# Adapted from qwen2.py
import logging
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.kernels.ops.attention.qk_norm_rope import qk_norm_rope
from sglang.kernels.ops.attention.table_qk_norm_rope import table_qk_norm_rope_
from sglang.srt.environ import envs
from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
from sglang.srt.layers.layer_boundary import (
    NORM_QUANT_READOUT,
    append_stages,
    declare_attn,
    declare_ffn,
)
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_activation import (
    aiter_prefers_bf16_input,
    aiter_takes_fp8_group_input,
    fp8_group_quant_fusion_allowed,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import RotaryEmbedding, get_rope
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import get_kv_layers, get_token_to_kv_pool
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
    supports_quantized_rl_reload,
)
from sglang.srt.models.qwen2 import Qwen2MLP as Qwen3MLP
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.utils import (
    apply_qk_norm_rope,
    can_use_jit_qk_norm_rope,
    maybe_enable_silu_fp4_quant_fusion,
)
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import (
    attention_backends,
    get_exec,
    get_parallel,
    get_spec,
    get_stream,
)
from sglang.srt.utils import (
    get_bool_env_var,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_npu,
    is_tpu,
    print_warning_once,
)

Qwen3Config = None

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_gfx95 = is_gfx95_supported()
_is_tpu = is_tpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

_has_fused_qk_norm_mrope = False
_has_fused_qk_norm_rope = False
if _use_aiter:
    from aiter import dtypes as aiter_dtypes

    try:
        from aiter import fused_qk_norm_mrope_3d_cache_pts_quant_shuffle

        _has_fused_qk_norm_mrope = True
        logger.info("aiter fused_qk_norm_mrope_3d kernel available")
    except ImportError:
        pass
    try:
        from aiter import fused_qk_norm_rope_cache_pts_quant_shuffle

        _has_fused_qk_norm_rope = True
        logger.info("aiter fused_qk_norm_rope kernel available")
    except ImportError:
        pass

if _is_npu:
    from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_rope

    from sglang.srt.hardware_backend.npu.cmo import get_cmo_stream, wait_cmo_stream

if _is_tpu:
    from sglang.srt.hardware_backend.tpu.fused_mlp import (
        TpuFusedMlp,
        mlp_geometry_ok,
    )
    from sglang.srt.hardware_backend.tpu.fused_mlp_ln import TpuFusedMlpLn
    from sglang.srt.hardware_backend.tpu.fused_qkv_ln import TpuFusedQkvLn
    from sglang.srt.hardware_backend.tpu.fused_qkv_rope import (
        TpuFusedQkvRope,
        kernel_geometry_ok,
    )
    from sglang.srt.hardware_backend.tpu.oproj_res import (
        TpuOprojRes,
        oproj_geometry_ok,
    )


@lru_cache(None)
def _warn_oproj_needs_fused_mlp_ln() -> None:
    """Say once per process that the o_proj flag is being ignored.

    The mismatch is a deployment config error and every layer would report it;
    ``print_warning_once`` is a plain ``logger.warning`` despite its name, so the
    once-ness has to come from here.
    """
    print_warning_once(
        "SGLANG_TPU_USE_FUSED_OPROJ needs SGLANG_TPU_USE_FUSED_MLP_LN=1: the "
        "fused o_proj writes the raw hidden + residual sum, which only the "
        "norm-fused MLP consumes. Ignoring it."
    )


def aiter_fused_qk_norm_rope_set_kv(
    qkv: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    rotary_emb: RotaryEmbedding,
    positions: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> torch.Tensor:
    """QK-norm + RoPE (1D, or 3D mRoPE for MRotaryEmbedding) + KV cache write in
    one aiter HIP kernel.

    ``qkv`` is the (num_tokens, (num_heads + 2 * num_kv_heads) * head_dim) QKV
    projection; ``k_cache`` / ``v_cache`` are the NHD (slots, num_kv_heads,
    head_dim) pool buffers, or the vectorized_5d ones, which are aiter's shuffle
    layout, in ``qkv``'s dtype or ``aiter.dtypes.fp8``. An fp8 cache stores
    k / k_scale and v / v_scale; other caches ignore the scales.
    ``k_scale`` / ``v_scale`` must be CPU tensors: the kernel reads them on the
    host. Returns q as (num_tokens, num_heads * head_dim).
    """
    num_tokens = qkv.shape[0]
    q_out = torch.empty(
        num_tokens, num_heads, head_dim, dtype=qkv.dtype, device=qkv.device
    )
    # A vectorized_5d K buffer is (blocks, heads, head_dim // x, block_size, x).
    shuffle = k_cache.dim() == 5
    kernel_kwargs = dict(
        qkv=qkv.view(num_tokens, -1, head_dim),
        qw=q_norm.weight,
        kw=k_norm.weight,
        cos_sin=rotary_emb.cos_sin_cache_as(qkv.dtype),
        positions=positions,
        num_tokens=num_tokens,
        num_heads_q=num_heads,
        num_heads_k=num_kv_heads,
        num_heads_v=num_kv_heads,
        head_size=head_dim,
        is_neox_style=rotary_emb.is_neox_style,
        eps=q_norm.variance_epsilon,
        q_out=q_out,
        # The kernel reads cache strides as (block, token, head); an NHD buffer
        # is one-slot blocks addressed with block_size=0.
        k_cache=k_cache if shuffle else k_cache.unsqueeze(1),
        v_cache=v_cache if shuffle else v_cache.unsqueeze(1),
        slot_mapping=out_cache_loc,
        per_tensor_k_scale=k_scale,
        per_tensor_v_scale=v_scale,
        k_out=None,
        v_out=None,
        return_kv=False,
        use_shuffle_layout=shuffle,
        block_size=k_cache.shape[3] if shuffle else 0,
        x=k_cache.shape[4] if shuffle else 0,
    )
    if isinstance(rotary_emb, MRotaryEmbedding):
        fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(
            **kernel_kwargs,
            mrope_section_=rotary_emb.mrope_section,
            is_interleaved=rotary_emb.mrope_interleaved,
        )
    else:
        fused_qk_norm_rope_cache_pts_quant_shuffle(**kernel_kwargs)
    return q_out.view(num_tokens, -1)


def _host_kv_scale(scale, scale_float) -> float:
    """Host value of a RadixAttention per-tensor KV scale. ``BaseKVCacheMethod``
    pairs its device scale with a float, ``load_kv_cache_scales`` sets a plain
    float, and without a scale the pool writes unscaled."""
    if scale is None:
        return 1.0
    if scale_float is not None:
        return float(scale_float)
    if isinstance(scale, (int, float)):
        return float(scale)
    # Reading it here would be a device-to-host sync, which graph capture forbids.
    raise ValueError(
        f"the fused KV write needs a host copy of the KV scale, got a "
        f"{type(scale).__name__} without one"
    )


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        start_layer: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = None,
        attention_bias: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        enable_prefill_qk_norm_rope: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.start_layer = start_layer
        self.total_num_heads = num_heads
        attn_tp_size = get_parallel().attn_tp_size

        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
            )
            if get_exec().deterministic.rl_on_policy_target is not None
            else {}
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            parallel_group="attn_tp",
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            parallel_group="attn_tp",
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )
        if (
            _is_npu
            and quant_config is not None
            and quant_config.get_name() == "modelslim"
        ):
            quant_config.init_kv_cache(self.attn, prefix, qkv_proj=self.qkv_proj)
        self.alt_stream = alt_stream
        self.use_jit_qk_norm_rope = can_use_jit_qk_norm_rope(
            self.q_norm,
            self.k_norm,
            self.rotary_emb,
            self.head_dim,
            torch.get_default_dtype(),
        )

        # Qwen3 guarantees matching norm eps and full-head NeoX RoPE; these
        # CUDA head sizes also select the non-fallback RoPE implementation.
        self.use_table_qk_norm_rope = (
            _is_cuda
            and get_exec().kernel.enable_fused_qk_norm_rope
            and not get_exec().deterministic.enable_deterministic_inference
            and not envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
            and get_exec().deterministic.rl_on_policy_target is None
            and get_exec().graph.cuda_graph_config.prefill.tc_compiler != "inductor"
            and (
                type(self.rotary_emb) is RotaryEmbedding
                or (
                    # At head_dim 256, triton_mrope_fused's BF16 rounding differs.
                    type(self.rotary_emb) is MRotaryEmbedding
                    and self.rotary_emb.mrope_section is not None
                    and self.head_dim in (64, 128)
                )
            )
            and self.head_dim in (64, 128, 256)
        )
        self.table_rope_positions_ndim = (
            2 if type(self.rotary_emb) is MRotaryEmbedding else 1
        )

        # Only the ordinary dense-Qwen3 caller opts in; inherited callers stay native.
        self.use_prefill_qk_norm_rope = (
            enable_prefill_qk_norm_rope
            and self.use_table_qk_norm_rope
            and type(self) is Qwen3Attention
            and type(self.rotary_emb) is RotaryEmbedding
            and quant_config is None
            and attn_tp_size == 1
            and self.head_dim == 128
            and torch.cuda.get_device_capability()[0] == 9
            and attention_backends()[0] in ("fa3", "flashinfer")
            and get_spec().speculative_algorithm is None
            and self.rotary_emb.is_neox_style
            and self.rotary_emb.rotary_dim == self.head_dim
            and self.q_norm.variance_epsilon == self.k_norm.variance_epsilon
        )

        self.use_fused_qk_norm_mrope = (
            _has_fused_qk_norm_mrope
            and isinstance(self.rotary_emb, MRotaryEmbedding)
            and getattr(self.rotary_emb, "mrope_section", None) is not None
        )
        # The 1D kernel is instantiated for these head sizes, indexes
        # cos_sin_cache as [pos, rotary_dim] and takes one eps for both norms;
        # scaled-RoPE subclasses stay on the unfused path.
        self.use_fused_qk_norm_rope = (
            _has_fused_qk_norm_rope
            and type(self.rotary_emb) is RotaryEmbedding
            and self.rotary_emb.rotary_dim == self.head_dim
            and self.head_dim in (64, 128, 256, 512)
            and self.q_norm.variance_epsilon == self.k_norm.variance_epsilon
        )
        # Built on first use: the KV scales are set after __init__, by
        # process_weights_after_loading or load_kv_cache_scales.
        self._fused_kv_scales: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        self._tpu_fused_qkv_rope = self._build_tpu_fused_qkv_rope()
        # The norm-fused variant is built by the decoder layer, which owns the
        # ``input_layernorm`` the kernel needs (``_attach_input_layernorm``).
        self._tpu_fused_qkv_ln = None
        # Likewise built on demand: the o_proj+residual op only makes sense with
        # the norm-fused MLP, which the decoder layer builds after this runs.
        self._tpu_oproj_res = None

    def _tpu_fused_geometry_ok(self) -> bool:
        """Whether the fused qkv kernel can serve this layer's geometry at all.

        The kernel bakes the head layout, so a layer that does not match exactly
        must never reach it.
        """
        if isinstance(self.rotary_emb, MRotaryEmbedding):
            return False
        if not self.rotary_emb.is_neox_style:
            return False
        if self.rotary_emb.rotary_dim != self.head_dim:
            return False
        if kernel_geometry_ok(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        ):
            return True
        logger.warning(
            "TPU fused qkv/rope disabled for layer %s: head geometry "
            "(H=%d, Hq=%d, Hkv=%d, D=%d) is outside the kernel's contract",
            self.attn.layer_id,
            self.hidden_size,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )
        return False

    def _build_tpu_fused_qkv_rope(self):
        """The TPU fused qkv/norm/RoPE/cache-write op, or None to keep torch.

        Decided once, here, because every condition is a property of the layer's
        geometry.  The per-forward checks (dtype, cross attention, sliding
        window) live at the call site.
        """
        if not (_is_tpu and envs.SGLANG_TPU_USE_FUSED_QKV_ROPE.get()):
            return None
        if not self._tpu_fused_geometry_ok():
            return None
        return TpuFusedQkvRope(
            qkv_proj=self.qkv_proj,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

    def _attach_input_layernorm(self, input_layernorm) -> None:
        """Build the norm-fused op against the layer's pre-attention norm.

        The kernel takes ``hidden + residual`` un-normalized and applies the norm
        itself, so it needs that norm's weight and epsilon -- which belong to the
        decoder layer, not to the attention.  When it is built the projection-only
        op is dropped: this one does everything it does, and building both would
        compile two kernels where one will ever run.
        """
        self._tpu_fused_qkv_ln = self._build_tpu_fused_qkv_ln(input_layernorm)
        if self._tpu_fused_qkv_ln is not None:
            self._tpu_fused_qkv_rope = None

    def _build_tpu_fused_qkv_ln(self, input_layernorm):
        """The input-norm-fused op, or None to keep the torch chain."""
        if not (_is_tpu and envs.SGLANG_TPU_USE_FUSED_QKV_LN.get()):
            return None
        if not self._tpu_fused_geometry_ok():
            return None
        return TpuFusedQkvLn(
            qkv_proj=self.qkv_proj,
            input_layernorm=input_layernorm,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            ln_eps=input_layernorm.variance_epsilon,
        )

    def _attach_tpu_oproj_res(self) -> None:
        """Build the ``o_proj`` + residual op, or leave it None.

        Called by the decoder layer once the MLP ops have resolved, and only when
        the norm-fused MLP was built: this op writes the raw ``hidden +
        residual`` sum, which is exactly that MLP's input.  Any other MLP path
        runs ``prepare_mlp`` over the attention output, and would add the
        residual to an already-summed hidden state.
        """
        self._tpu_oproj_res = self._build_tpu_oproj_res()

    def _build_tpu_oproj_res(self):
        """The fused o_proj + residual op, or None to keep the torch chain.

        Decided once, here: device, weight layout and tensor parallelism are all
        fixed for the layer's lifetime.  Checked against the actual weight rather
        than a quant config, because a packed/quantized ``o_proj`` would change
        both the dtype and the shape the kernel is built for.
        """
        if not (_is_tpu and envs.SGLANG_TPU_USE_FUSED_OPROJ.get()):
            return None
        # Read the geometry off the weight itself: a packed/quantized ``o_proj``
        # has neither the dtype nor the shape the kernel is built for, and the
        # head count would not say so.
        weight = self.o_proj.weight
        if weight.dtype != torch.bfloat16 or weight.dim() != 2:
            return None
        weight_hidden, input_size = weight.shape
        if weight_hidden != self.hidden_size or self.o_proj.bias is not None:
            # The kernel's ABI has no bias term.
            return None
        if get_parallel().tp_size != 1:
            # o_proj is row-parallel: above TP=1 it all-reduces the partial
            # output, which the kernel does not do.
            return None
        if not oproj_geometry_ok(
            hidden_size=self.hidden_size, input_size=input_size
        ):
            logger.warning(
                "TPU fused o_proj disabled for layer %s: geometry (H=%d, K=%d) "
                "is outside the kernel's contract",
                self.attn.layer_id,
                self.hidden_size,
                input_size,
            )
            return None
        return TpuOprojRes(
            o_proj=self.o_proj,
            hidden_size=self.hidden_size,
            input_size=input_size,
        )

    def _tpu_fused_batch_ok(self, hidden_states, forward_batch) -> bool:
        """Conditions both fused prepares share: the layout they consume, nothing
        else -- bf16 activations, a plain self-attention layer with no sliding
        window, and a paged-cache slot per token."""
        return (
            hidden_states.dtype == torch.bfloat16
            and not self.attn.is_cross_attention
            and self.attn.sliding_window_size < 0
            and forward_batch.out_cache_loc is not None
            and get_exec().deterministic.rl_on_policy_target is None
        )

    def _tpu_fused_ok(self, hidden_states, forward_batch) -> bool:
        """Per-forward gate for the projection-only op."""
        return self._tpu_fused_qkv_rope is not None and self._tpu_fused_batch_ok(
            hidden_states, forward_batch
        )

    def _tpu_fused_ln_ok(
        self, hidden_states, forward_batch, post_residual_addition
    ) -> bool:
        """Per-forward gate for the norm-fused op, asked by the decoder layer.

        The layer asks because it has to hand over an *unnormalized* sum instead
        of running ``prepare_attn``; the attention is told the answer rather than
        re-deriving it, so the two can never disagree about who normalized.
        """
        return (
            self._tpu_fused_qkv_ln is not None
            and post_residual_addition is None
            and self._tpu_fused_batch_ok(hidden_states, forward_batch)
        )

    def forward_prepare_tpu(
        self, positions, hidden_states, forward_batch, *, with_input_norm=False
    ):
        """Fused [input norm] + qkv_proj + qk-norm + RoPE + paged-cache write.

        ``with_input_norm`` means ``hidden_states`` is the layer's raw
        ``hidden + residual`` and the kernel normalizes it; otherwise it is
        already normalized (the projection-only variant).

        K/V are in the cache when this returns, so the attention backend must not
        write them again: the caller passes save_kv_cache=False and (q, None,
        None), exactly like the aiter fused-mRoPE path.
        """
        # The torch RoPE path co-locates the cache on first use; bypassing it
        # leaves cos_sin_cache on its init device (CPU).  Same call, same
        # effect: move it to the activation's device and dtype.
        logger.info_once(
            "TPU fused qkv/rope engaged: hidden=%d, heads=%d/%d, head_dim=%d, "
            "input_norm=%s",
            hidden_states.shape[1],
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            with_input_norm,
        )
        self.rotary_emb._match_cos_sin_cache_dtype(hidden_states)
        k_cache, v_cache = get_token_to_kv_pool().get_kv_buffer(self.attn.layer_id)
        op = self._tpu_fused_qkv_ln if with_input_norm else self._tpu_fused_qkv_rope
        q = op(
            hidden_states=hidden_states,
            positions=positions,
            cos_sin_cache=self.rotary_emb.cos_sin_cache,
            out_cache_loc=forward_batch.out_cache_loc,
            k_cache=k_cache,
            v_cache=v_cache,
            eps=self.q_norm.variance_epsilon,
        )
        return q, None, None

    def _aiter_fused_kv_scales(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(k_scale, v_scale) for the fused KV write as CPU float32 tensors. The
        kernel reads them on the host at launch, so a device tensor would sync
        and break graph capture, and a captured graph keeps the values it was
        captured with."""
        if self._fused_kv_scales is None:
            k_scale = _host_kv_scale(self.attn.k_scale, self.attn.k_scale_float)
            v_scale = _host_kv_scale(self.attn.v_scale, self.attn.v_scale_float)
            self._fused_kv_scales = (
                torch.tensor(k_scale, dtype=torch.float32, device="cpu"),
                torch.tensor(v_scale, dtype=torch.float32, device="cpu"),
            )
        return self._fused_kv_scales

    def forward_prepare_native(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if (
            self.use_table_qk_norm_rope
            and forward_batch.forward_mode.is_decode()
            and qkv.dtype == torch.bfloat16
            and self.q_norm.weight.dtype == torch.bfloat16
            and self.k_norm.weight.dtype == torch.bfloat16
            and positions.ndim == self.table_rope_positions_ndim
        ):
            axis_map = None
            if positions.ndim == 2:
                axis_map = self.rotary_emb.axis_map
                self.rotary_emb._match_cos_sin_cache_dtype(qkv)
            table_qk_norm_rope_(
                qkv,
                positions,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rotary_emb.cos_sin_cache,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.q_norm.variance_epsilon,
                match_cuda=True,
                axis_map=axis_map,
            )
        elif (
            self.use_prefill_qk_norm_rope
            and forward_batch.forward_mode == ForwardMode.EXTEND
            and not torch.compiler.is_compiling()
            and qkv.is_cuda
            and qkv.is_contiguous()
            and qkv.dtype in (torch.bfloat16, torch.float16)
            and self.q_norm.weight.dtype == qkv.dtype
            and self.k_norm.weight.dtype == qkv.dtype
            and self.rotary_emb.cos_sin_cache.dtype == torch.float32
            and positions.is_contiguous()
            and positions.dtype in (torch.int32, torch.int64)
        ):
            q = qk_norm_rope(
                q=q,
                k=k,
                q_weight=self.q_norm.weight,
                k_weight=self.k_norm.weight,
                cos_sin_cache=self.rotary_emb.cos_sin_cache,
                positions=positions,
                head_dim=self.head_dim,
                eps=self.q_norm.variance_epsilon,
            )
        else:
            q, k = apply_qk_norm_rope(
                q=q,
                k=k,
                q_norm=self.q_norm,
                k_norm=self.k_norm,
                rotary_emb=self.rotary_emb,
                positions=positions,
                head_dim=self.head_dim,
                alt_stream=self.alt_stream,
                use_jit=self.use_jit_qk_norm_rope,
            )
        return q, k, v

    def forward_prepare_npu(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn.layer_id == self.start_layer:
            self.rotary_emb.get_cos_sin_with_position(positions)
        q, k, v = split_qkv_rmsnorm_rope(
            qkv,
            self.rotary_emb.position_sin,
            self.rotary_emb.position_cos,
            self.q_size,
            self.kv_size,
            self.head_dim,
            eps=self.q_norm.variance_epsilon,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            q_bias=getattr(self.q_norm, "bias", None),
            k_bias=getattr(self.k_norm, "bias", None),
        )
        return q, k, v

    def forward_prepare_aiter_fused(
        self, positions, hidden_states, forward_batch, k_cache, v_cache
    ):
        """Decode-only fused path; KV is already in the paged cache when this
        returns, so the caller must pass save_kv_cache=False to attn."""
        qkv, _ = self.qkv_proj(hidden_states)
        q = aiter_fused_qk_norm_rope_set_kv(
            qkv,
            self.q_norm,
            self.k_norm,
            self.rotary_emb,
            positions,
            k_cache,
            v_cache,
            forward_batch.out_cache_loc,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            *self._aiter_fused_kv_scales(),
        )
        return q, None, None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        fused_input_norm: bool = False,
        oproj_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attention, then ``o_proj``.

        ``oproj_residual`` is the layer's residual: when it is given, the fused
        op adds it to the projection on chip and what comes back is the raw
        ``hidden + residual`` sum, not just the projection.  The layer decides
        and only passes it when the norm-fused MLP will consume that sum.
        """
        if get_exec().deterministic.rl_on_policy_target is not None:
            hidden_states = hidden_states.bfloat16()

        save_kv_cache = True
        use_aiter_fused = False
        can_try_aiter_fused = (
            (self.use_fused_qk_norm_mrope or self.use_fused_qk_norm_rope)
            and forward_batch.forward_mode.is_decode()
            and get_exec().deterministic.rl_on_policy_target is None
        )
        if can_try_aiter_fused:
            k_cache, v_cache = get_kv_layers().read(self.attn.layer_id).kv()
            # The fused write targets the NHD layout or the vectorized_5d one, in
            # the activation dtype or the one fp8 dtype the kernel quantizes to.
            # A fused-norm (fp8, scale) input still yields a bf16 qkv.
            act_dtype = (
                torch.bfloat16
                if isinstance(hidden_states, tuple)
                else hidden_states.dtype
            )
            use_aiter_fused = k_cache.dim() in (3, 5) and (
                k_cache.dtype in (act_dtype, aiter_dtypes.fp8)
            )

        if fused_input_norm:
            # The layer handed over hidden + residual un-normalized; the kernel
            # applies the input norm, so it must not be normalized here too.
            q, k, v = self.forward_prepare_tpu(
                positions, hidden_states, forward_batch, with_input_norm=True
            )
            save_kv_cache = False
        elif self._tpu_fused_ok(hidden_states, forward_batch):
            q, k, v = self.forward_prepare_tpu(
                positions, hidden_states, forward_batch
            )
            save_kv_cache = False
        elif use_aiter_fused:
            q, k, v = self.forward_prepare_aiter_fused(
                positions, hidden_states, forward_batch, k_cache, v_cache
            )
            save_kv_cache = False
        elif not _is_npu:
            q, k, v = self.forward_prepare_native(
                positions=positions,
                hidden_states=hidden_states,
                # use fused kernel for decode
                forward_batch=forward_batch,
            )
        else:
            q, k, v = self.forward_prepare_npu(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        if get_exec().deterministic.rl_on_policy_target is not None:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)

        attn_output = self.attn(q, k, v, forward_batch, save_kv_cache=save_kv_cache)
        # A fused qkv op hands back its whole padded query tile (the paged
        # attention kernel needs a padded buffer anyway), so the attention
        # output carries the same padding; this trims it to the batch's rows.
        # Nothing else pads, so on every other path the shapes already agree
        # and this is a single int comparison.
        num_tokens = hidden_states.shape[0]
        if attn_output.shape[0] != num_tokens:
            attn_output = attn_output[:num_tokens]
        if oproj_residual is not None:
            # Only the layer knows whether the norm-fused MLP is live, and it
            # builds this op only in that case, so a residual here implies the op.
            assert self._tpu_oproj_res is not None
            return self._tpu_oproj_res(
                attn_output=attn_output, residual=oproj_residual
            )
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_id: int = 0,
        start_layer: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        if (
            hasattr(config, "rope_parameters")
            and config.rope_parameters
            and "rope_theta" in config.rope_parameters
        ):
            rope_theta = config.rope_parameters["rope_theta"]
            rope_scaling = config.rope_parameters
        else:
            rope_theta = getattr(config, "rope_theta", 1000000)
            rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            start_layer=start_layer,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
            enable_prefill_qk_norm_rope=(
                type(self) is Qwen3DecoderLayer and config.model_type == "qwen3"
            ),
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            reduce_results=False,
        )
        maybe_enable_silu_fp4_quant_fusion(self.mlp)

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
                override_orig_dtype=torch.float32,
                fp32_residual=True,
            )
            if get_exec().deterministic.rl_on_policy_target is not None
            else {}
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
        )

        attn, ffn = self.stage_facts(config, layer_id)
        self.attn_boundary, self.ffn_boundary = append_stages(
            (attn, self.input_layernorm),
            (ffn, self.post_attention_layernorm),
        )
        # Only aiter on gfx95 needs detection after weight loading.
        # Other platforms can resolve the empty formats before tracing forward.
        self._decode_quant_formats = None if (_use_aiter and _is_gfx95) else ("", "")

        self.self_attn._attach_input_layernorm(self.input_layernorm)
        # The norm-fused op supersedes the plain one, exactly as in the attention:
        # it does everything the plain op does, and building both would compile
        # two kernels where one will ever run.
        self._tpu_fused_mlp_ln = self._build_tpu_fused_mlp_ln(
            quant_config=quant_config,
            intermediate_size=config.intermediate_size,
            layer_id=layer_id,
        )
        self._tpu_fused_mlp = (
            None
            if self._tpu_fused_mlp_ln is not None
            else self._build_tpu_fused_mlp(
                quant_config=quant_config,
                intermediate_size=config.intermediate_size,
                layer_id=layer_id,
            )
        )
        if self._tpu_fused_mlp_ln is not None:
            # The o_proj op hands its raw sum straight to the norm-fused MLP, so
            # it exists only alongside it.
            self.self_attn._attach_tpu_oproj_res()
        elif _is_tpu and envs.SGLANG_TPU_USE_FUSED_OPROJ.get():
            _warn_oproj_needs_fused_mlp_ln()

    @staticmethod
    def stage_facts(config: Qwen3Config, layer_id: int):
        """The stages a Qwen3 layer declares, from the config alone: the
        model's shared declaration function, which the layer declares with
        too (see make_layers)."""
        return (
            declare_attn(),
            declare_ffn(sparse=False, next_layer_sparse=False, read=NORM_QUANT_READOUT),
        )

    def _quant_formats(self, forward_batch: ForwardBatch, positions: torch.Tensor):
        """quant_format for the attention and FFN input norms: "fp8" fuses the
        1x128 fp8 quant of qkv_proj / gate_up_proj into the norm (decode only,
        unless the linear runs this batch's rows faster on a bf16 input)."""
        if not forward_batch.forward_mode.is_decode():
            return "", ""
        linears = (self.self_attn.qkv_proj, self.mlp.gate_up_proj)
        if self._decode_quant_formats is None:
            self._decode_quant_formats = tuple(
                "fp8"
                if _use_aiter
                and _is_gfx95
                and aiter_takes_fp8_group_input(linear)
                # The GEMM returns bf16 for a pre-quantized input.
                and linear.params_dtype == torch.bfloat16
                else ""
                for linear in linears
            )
        if not any(self._decode_quant_formats) or not fp8_group_quant_fusion_allowed():
            return "", ""
        # The layer input may be an opaque OwedOutput; positions has one entry
        # per token (in its last dim under M-RoPE).
        num_tokens = positions.shape[-1]
        # A read that the input moves after (attention DP, scattered rows)
        # cannot hand on the quantized input.
        return tuple(
            (
                quant
                if quant
                and boundary.reads_quantized(forward_batch)
                and not aiter_prefers_bf16_input(linear, num_tokens)
                else ""
            )
            for quant, boundary, linear in zip(
                self._decode_quant_formats,
                (self.attn_boundary, self.ffn_boundary),
                linears,
            )
        )

    def _tpu_fused_mlp_common_ok(self, *, quant_config, intermediate_size, layer_id):
        """Gates the plain and the norm-fused MLP op share."""
        if quant_config is not None:
            return False
        if get_parallel().tp_size != 1:
            # Two reasons, and they are the same gate.  down_proj is row-parallel:
            # above TP=1 it owns a slice of the I reduction and all-reduces
            # afterwards, which the kernel does not do.  And prepare_mlp only
            # reduces to `_simple` -- a residual add plus the norm, nothing else
            # -- while attn_tp_size is 1, so the norm-fused hand-off below is only
            # equivalent to it at TP=1.
            return False
        if not mlp_geometry_ok(
            hidden_size=self.hidden_size, intermediate_size=intermediate_size
        ):
            logger.warning(
                "TPU fused MLP disabled for layer %s: geometry (H=%d, I=%d) is "
                "outside the kernel's contract",
                layer_id,
                self.hidden_size,
                intermediate_size,
            )
            return False
        return True

    def _build_tpu_fused_mlp_ln(
        self, *, quant_config, intermediate_size: int, layer_id: int
    ):
        """The post-attention-norm-fused op, or None to fall back.

        The kernel takes the raw ``hidden + residual`` and applies
        ``post_attention_layernorm`` itself, so it needs that norm's weight and
        epsilon; the layer then skips ``prepare_mlp`` instead of running it.
        """
        if not (_is_tpu and envs.SGLANG_TPU_USE_FUSED_MLP_LN.get()):
            return None
        if not self._tpu_fused_mlp_common_ok(
            quant_config=quant_config,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
        ):
            return None
        return TpuFusedMlpLn(
            gate_up_proj=self.mlp.gate_up_proj,
            down_proj=self.mlp.down_proj,
            post_attention_layernorm=self.post_attention_layernorm,
            hidden_size=self.hidden_size,
            intermediate_size=intermediate_size,
            ln_eps=self.post_attention_layernorm.variance_epsilon,
        )

    def _tpu_fused_mlp_ln_ok(self, hidden_states) -> bool:
        """Per-forward gate for the norm-fused op, asked by ``forward``.

        The layer asks because it has to hand over an *unnormalized* sum instead
        of running ``prepare_mlp``; the answer also picks the op, so the two can
        never disagree about who normalized.  Everything else is fixed at
        construction -- ``prepare_mlp`` has no ``post_residual_addition`` and no
        batch-dependent branch under TP=1 -- so only the activation dtype and the
        RL fp32-residual variant vary per forward.
        """
        return (
            self._tpu_fused_mlp_ln is not None
            and hidden_states.dtype == torch.bfloat16
            and get_exec().deterministic.rl_on_policy_target is None
        )

    def _build_tpu_fused_mlp(
        self, *, quant_config, intermediate_size: int, layer_id: int
    ):
        """The fused gate_up/silu_mul/down op, or None to keep the torch chain.

        Decided once, here: device, quantization and tensor parallelism are all
        fixed for the layer's lifetime.  ``prepare_mlp`` normalizes the hidden
        state before the MLP runs, which is exactly what the kernel consumes, so
        no norm is fused in and the layer keeps its ``prepare_mlp`` boundary.
        ``_build_tpu_fused_mlp_ln`` is the variant that folds that norm in; when
        it is built this one is not.
        """
        if not (_is_tpu and envs.SGLANG_TPU_USE_FUSED_MLP.get()):
            return None
        if not self._tpu_fused_mlp_common_ok(
            quant_config=quant_config,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
        ):
            return None
        return TpuFusedMlp(
            gate_up_proj=self.mlp.gate_up_proj,
            down_proj=self.mlp.down_proj,
            hidden_size=self.hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        post_residual_addition: Optional[torch.Tensor] = None,
        capture_output=None,
    ) -> torch.Tensor:
        attn_quant, ffn_quant = self._quant_formats(forward_batch, positions)
        if post_residual_addition is not None:
            # The fused fp8 norm does not apply the addition.
            attn_quant = ""
        # Asked before attention, not after: when the fused o_proj runs it emits
        # the raw ``hidden + residual`` sum, and only the norm-fused MLP consumes
        # that.  Deciding afterwards would let the sum be formed and then have
        # the FFN prepare add the residual to it a second time.  Attention cannot
        # change the answer -- the gate varies only on the op being built, the
        # activation dtype and the RL target, none of which attention touches --
        # so asking early asks the same question.  Layer 0 has no residual and
        # keeps the plain ``o_proj``.
        fused_mlp_ln = self._tpu_fused_mlp_ln_ok(hidden_states)
        fused_oproj = (
            fused_mlp_ln
            and self.self_attn._tpu_oproj_res is not None
            and residual_batch.written_residual(forward_batch) is not None
        )
        # Self Attention
        # quant_format goes only to a read that quantizes; other paths are called
        # as for an unquantized model.
        fused_input_norm = (
            capture_output is None
            and self.self_attn._tpu_fused_ln_ok(
                hidden_states, forward_batch, post_residual_addition
            )
        )
        if fused_input_norm:
            # The kernel normalizes on chip, so it is handed the raw sum, which
            # is also the residual this layer carries forward: fold the stream
            # instead of letting prepare apply the norm.
            hidden_states = residual_batch.fold(hidden_states, forward_batch)
        else:
            hidden_states = self.attn_boundary.prepare(
                hidden_states,
                forward_batch,
                post_residual_addition=post_residual_addition,
                capture=capture_output,
                **({"quant_format": attn_quant} if attn_quant else {}),
            )
        num_tokens = (
            hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        ).shape[0]
        if num_tokens != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                fused_input_norm=fused_input_norm,
                oproj_residual=(
                    residual_batch.written_residual(forward_batch)
                    if fused_oproj
                    else None
                ),
            )

        # Fully Connected
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        if fused_oproj:
            # The o_proj op already formed the sum the FFN prepare would have:
            # what came back is both the MLP input and the residual to carry.
            residual_batch.set_written(hidden_states, forward_batch)
        elif fused_mlp_ln:
            # Same hand-off as the attention above: the norm-fused kernel takes
            # the raw sum, which is also the residual this layer carries.
            hidden_states = residual_batch.fold(hidden_states, forward_batch)
        else:
            hidden_states = self.ffn_boundary.prepare(
                hidden_states,
                forward_batch,
                **({"quant_format": ffn_quant} if ffn_quant else {}),
                cache=(
                    [self.mlp.gate_up_proj.weight, self.mlp.down_proj.weight]
                    if _is_npu
                    and check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
                    and (
                        hasattr(self.mlp.gate_up_proj, "weight")
                        and hasattr(self.mlp.down_proj, "weight")
                    )
                    else None
                ),
            )
        # Keyed on the gate, not on "is the op built": when the gate declines,
        # the FFN prepare has already normalized and the norm-fused kernel would
        # normalize a second time.
        fused_mlp = self._tpu_fused_mlp_ln if fused_mlp_ln else self._tpu_fused_mlp
        if fused_mlp is not None:
            hidden_states = fused_mlp(x=hidden_states)
        else:
            hidden_states = self.mlp(hidden_states, forward_batch=forward_batch)
        if _is_npu and get_cmo_stream():
            wait_cmo_stream()
        hidden_states = self.ffn_boundary.finish(hidden_states, forward_batch)
        return hidden_states


class Qwen3Model(Qwen2Model):
    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        alt_stream = get_stream("alt") if _is_cuda else None
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=Qwen3DecoderLayer,
            alt_stream=alt_stream,
        )

    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        if self.pp_group.is_first_rank:
            hidden_states = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
            residual_batch.start(forward_batch)
        else:
            hidden_states = self.layers[self.start_layer].attn_boundary.from_pp(
                pp_proxy_tensors, forward_batch
            )
        aux_hidden_states = AuxHiddenStateList()
        for i in range(self.start_layer, self.end_layer):
            hidden_states = self.layers[i](
                positions,
                hidden_states,
                forward_batch,
                capture_output=aux_hidden_states.capture
                if i in self.layers_to_capture
                else None,
            )
        if not self.pp_group.is_last_rank:
            return residual_batch.to_pp(hidden_states, forward_batch)
        hidden_states = residual_batch.final_norm(
            hidden_states, forward_batch, self.norm, skip_empty=True
        )
        return (
            (hidden_states, aux_hidden_states) if aux_hidden_states else hidden_states
        )


class Qwen3ForCausalLM(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_parallel().pp_group
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # handle the lm head on different pp ranks
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    use_attn_tp_group=get_parallel().enable_dp_lm_head,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    @supports_quantized_rl_reload
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if not name.startswith("model.") and (
                name.startswith("layers.")
                or name.startswith("embed_tokens.")
                or name.startswith("norm.")
            ):
                name = add_prefix(name, "model")

            if name == "model.embed_tokens.weight":
                if self.pp_group.is_last_rank and self.config.tie_word_embeddings:
                    if "lm_head.weight" in params_dict:
                        param = params_dict["lm_head.weight"]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            if "scale" in name or (_is_npu and ".fa_" in name):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        if hasattr(self.model.embed_tokens, "weight"):
            del self.model.embed_tokens.weight
        if hasattr(self.lm_head, "weight"):
            del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        current_platform.empty_cache()
        current_platform.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def set_dflash_layers_to_capture(self, layer_ids: List[int]):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )

        self.capture_aux_hidden_states = True
        # SGLang captures "before layer i". To capture the hidden state after target
        # layer `k` (HF-style), we capture before layer `k + 1`.
        self.model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = Qwen3ForCausalLM
