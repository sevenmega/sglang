# Adapted from qwen2.py
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
)
from sglang.srt.environ import envs
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.qwen2 import Qwen2MLP as Qwen3MLP
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_exec, get_parallel, get_stream
from sglang.srt.utils import add_prefix, get_bool_env_var, is_cuda, is_hip, is_npu, is_tpu

Qwen3Config = None

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_tpu = is_tpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

_has_fused_qk_norm_mrope = False
if _use_aiter:
    try:
        from aiter import fused_qk_norm_mrope_3d_cache_pts_quant_shuffle

        _has_fused_qk_norm_mrope = True
        logger.info("aiter fused_qk_norm_mrope_3d kernel available")
    except ImportError:
        pass

if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope

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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.start_layer = start_layer
        self.tp_size = get_parallel().tp_size
        self.total_num_heads = num_heads
        attn_tp_rank = get_parallel().attn_tp_rank
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
        self.tp_rank = get_parallel().tp_rank

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
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
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
        self.alt_stream = alt_stream

        self.use_fused_qk_norm_mrope = (
            _has_fused_qk_norm_mrope
            and isinstance(self.rotary_emb, MRotaryEmbedding)
            and getattr(self.rotary_emb, "mrope_section", None) is not None
        )
        if self.use_fused_qk_norm_mrope:
            # Scale tensors MUST stay on CPU: the C++ kernel uses .item<float>()
            # which triggers hipMemcpy D2H + sync on CUDA tensors, breaking graph capture.
            # Explicit device='cpu' is required because SGLang constructs models inside
            # a `with torch.device('cuda'):` context that changes the default device.
            self._fused_k_scale = torch.tensor(1.0, dtype=torch.float32, device="cpu")
            self._fused_v_scale = torch.tensor(1.0, dtype=torch.float32, device="cpu")

        self._tpu_fused_qkv_rope = self._build_tpu_fused_qkv_rope()
        # The norm-fused variant is built by the decoder layer, which owns the
        # ``input_layernorm`` the kernel needs (``_attach_input_layernorm``).
        self._tpu_fused_qkv_ln = None

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

    def forward_prepare_native(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        q, k = self.rotary_emb(positions, q, k)
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

    def forward_prepare_aiter_fused_mrope(
        self, positions, hidden_states, forward_batch
    ):
        """Fused QK-norm + 3D mRoPE + KV cache write for decode (ROCm/aiter).

        The fused HIP kernel replaces split → QK norm → mRoPE → cache write,
        so KV is already in the paged cache when this returns.
        Returns (q, None, None); caller must pass save_kv_cache=False to attn.
        """
        qkv, _ = self.qkv_proj(hidden_states)
        num_tokens = qkv.shape[0]

        qkv_3d = qkv.view(num_tokens, -1, self.head_dim)

        token_to_kv_pool = get_token_to_kv_pool()
        k_cache, v_cache = token_to_kv_pool.get_kv_buffer(self.attn.layer_id)
        slot_mapping = forward_batch.out_cache_loc

        cos_sin = self.rotary_emb.cos_sin_cache
        if cos_sin.dtype != qkv.dtype:
            cos_sin = cos_sin.to(dtype=qkv.dtype)

        q_out = torch.empty(
            num_tokens,
            self.num_heads,
            self.head_dim,
            dtype=qkv.dtype,
            device=qkv.device,
        )

        fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(
            qkv_3d,
            self.q_norm.weight,
            self.k_norm.weight,
            cos_sin,
            positions,
            num_tokens,
            self.num_heads,
            self.num_kv_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rotary_emb.is_neox_style,
            self.rotary_emb.mrope_section,
            self.rotary_emb.mrope_interleaved,
            self.q_norm.variance_epsilon,
            q_out,
            k_cache,
            v_cache,
            slot_mapping,
            self._fused_k_scale,
            self._fused_v_scale,
            None,
            None,
            False,
            False,
            0,
            0,
        )

        q = q_out.reshape(num_tokens, -1)
        return q, None, None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        fused_input_norm: bool = False,
    ) -> torch.Tensor:
        if get_exec().deterministic.rl_on_policy_target is not None:
            hidden_states = hidden_states.bfloat16()

        save_kv_cache = True
        use_aiter_fused = (
            self.use_fused_qk_norm_mrope
            and forward_batch.forward_mode.is_decode()
            and get_exec().deterministic.rl_on_policy_target is None
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
            q, k, v = self.forward_prepare_aiter_fused_mrope(
                positions, hidden_states, forward_batch
            )
            save_kv_cache = False
        elif not _is_npu:
            q, k, v = self.forward_prepare_native(
                positions=positions,
                hidden_states=hidden_states,
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
        output, _ = self.o_proj(attn_output)
        return output


def _fused_ln_residual(
    hidden_states: torch.Tensor, residual: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The residual half of ``prepare_attn`` / ``prepare_mlp``, norm left to a kernel.

    Reproduces ``RMSNorm.forward_native``'s arithmetic: the sum is formed in fp32
    and rounded once, so the residual stream stays bit-identical to the torch
    path's instead of drifting a bf16 ulp per layer.  ``residual is None`` (the
    first layer) and the empty batch mirror ``prepare_*``: the carried residual is
    exactly what came in.  ``post_residual_addition`` never reaches here: the
    fused attention path's gate requires it to be None, and ``prepare_mlp`` has
    no such argument.
    """
    if residual is None or hidden_states.shape[0] == 0:
        return hidden_states, hidden_states
    summed = (hidden_states.float() + residual.float()).to(hidden_states.dtype)
    return summed, summed


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
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

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

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=False,
            is_previous_layer_sparse=False,
            is_next_layer_sparse=False,
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )
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
        residual: Optional[torch.Tensor],
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        fused_input_norm = self.self_attn._tpu_fused_ln_ok(
            hidden_states, forward_batch, post_residual_addition
        )
        if fused_input_norm:
            # The kernel normalizes on chip, so it takes the raw sum -- which is
            # also the residual this layer carries forward.  Same hand-off as
            # prepare_attn, minus the norm.
            hidden_states, residual = _fused_ln_residual(hidden_states, residual)
        else:
            hidden_states, residual = self.layer_communicator.prepare_attn(
                hidden_states,
                residual,
                forward_batch,
                post_residual_addition=post_residual_addition,
            )
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                fused_input_norm=fused_input_norm,
            )

        # Fully Connected
        fused_mlp_ln = self._tpu_fused_mlp_ln_ok(hidden_states)
        if fused_mlp_ln:
            # Same hand-off as the attention above, for prepare_mlp: the kernel
            # normalizes on chip, so it takes the raw sum -- which is also the
            # residual this layer carries forward.
            hidden_states, residual = _fused_ln_residual(hidden_states, residual)
        else:
            hidden_states, residual = self.layer_communicator.prepare_mlp(
                hidden_states,
                residual,
                forward_batch,
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
        # prepare_mlp has already normalized and the norm-fused kernel would
        # normalize a second time.
        fused_mlp = self._tpu_fused_mlp_ln if fused_mlp_ln else self._tpu_fused_mlp
        if fused_mlp is not None:
            hidden_states = fused_mlp(x=hidden_states)
        else:
            hidden_states = self.mlp(hidden_states, forward_batch=forward_batch)
        if _is_npu and get_cmo_stream():
            wait_cmo_stream()
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual


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
        self.pp_group = get_pp_group()
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

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
        # decoder layer
        for i in range(start, end):
            layer = self.model.layers[i]
            forward_batch.hidden_states, forward_batch.residual = layer(
                positions,
                forward_batch.hidden_states,
                forward_batch,
                forward_batch.residual,
            )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

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
            if "scale" in name:
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
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

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
