"""TPU (Sophgo SG2260E) attention backend for SGLang.

The device only produces correct matmuls in bf16, its batched matmul (bmm) is
broken, and ``F.scaled_dot_product_attention`` is unimplemented. So instead of
SDPA this backend reuses :class:`TorchNativeAttnBackend`'s paged KV-cache gather
and per-request extend/decode plumbing, and replaces only the core attention
math. There are two implementations, tried in that order:

1. **Flash attention** (``SGLANG_TPU_USE_FLASH_ATTN``, on by default): the
   tilelang PPL online-softmax GQA kernel in
   ``tilelang.tpu.kernels.attention.flash_attention_gqa``, run one request at a
   time with the head-major fp16 layout the kernel expects. See
   :meth:`TpuAttnBackend._attend_flash` for the tiling/padding contract.
2. **Per-head 2D-matmul loop** (fallback, and used whenever 1 fails to build or
   launch):

    scores = (q_i @ k_i.tᵀ)·scale            # bf16 2D matmul, contiguous operands
    scores = scores.float() + causal_mask     # additive float mask, finite -1e9
    p      = softmax(scores).to(bf16)
    out_i  = p @ v_i                          # bf16 2D matmul

This matches a CPU-bf16 reference to <0.5% relative error (verified on hardware).
GQA (Qwen3: more query heads than kv heads) maps each query head to its kv group.
The causal mask is built on CPU (``triu`` and ``-inf`` misbehave on-device) with a
finite large-negative value and moved to the device.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Finite stand-in for -inf in the additive mask: -inf overflows on-device.
_MASK_NEG = -1e9
# fp16-safe stand-in for -inf: the flash kernel computes in fp16 internally, and
# -1e9 overflows fp16 (max ~65504). Anything <= this is treated as masked.
_FLASH_MASK_NEG = -30000.0

# Module-cached flash kernels keyed by (D, scaling). The kernel is shape-generic
# (dynamic B/Hq/Sq/Hkv/Skv), so one build per (head-dim, scale) serves every
# request; building invokes ppl-compile (seconds), so it runs at most once.
_FLASH_KERNELS: dict[tuple, object] = {}
# Set once we discover the flash kernel is unavailable (import/build failure) so
# we don't retry ppl-compile on every request.
_FLASH_DISABLED = False

# The PPL wrapper binds its stream/device in ``py_init_device`` (once, at kernel
# build) from ``TPU_VISIBLE_DEVICES``, but the torch_tpu runtime keeps its own
# "current device" and silently drifts it -- notably to a *stale* index -- after
# any device-tensor alloc or copy. The next PPL launch then dies in
# ``tpuRtSendLaunchKernelMsg`` with "stream and device mismatch!". Re-asserting
# the device through the runtime's own API immediately before each launch puts
# the two back in agreement. Resolved lazily so this module imports without the
# TPU runtime present.
_TPURT_SET_DEVICE = None


def _rt_device() -> int:
    """The visible device index the PPL wrapper's stream is bound to."""
    import os

    return int(os.environ.get("TPU_VISIBLE_DEVICES", "0").split(",")[0])


def _sync_rt_device() -> None:
    """Re-point the PPL runtime's current device at the one its stream uses.

    Must be called immediately before every PPL kernel launch, and after any
    torch_tpu device-tensor operation. A no-op if the runtime API is missing.
    """
    global _TPURT_SET_DEVICE
    if _TPURT_SET_DEVICE is None:
        try:
            import ctypes

            # The runtime is already mapped by torch_tpu (CDLL(None) searches
            # the global namespace), and re-opening the .so by path would pull
            # in libbmlib.so.0, which is not on the loader path here.
            lib = ctypes.CDLL(None, mode=ctypes.RTLD_GLOBAL)
            fn = lib.tpuRtSetDevice
            fn.argtypes = [ctypes.c_int]
            fn.restype = ctypes.c_int
            _TPURT_SET_DEVICE = fn
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[TPU] cannot re-assert the PPL runtime device (%s); "
                "repeat launches may fail.",
                exc,
            )
            _TPURT_SET_DEVICE = False  # tried and failed; don't retry
            return
    if _TPURT_SET_DEVICE is False:
        return
    _TPURT_SET_DEVICE(_rt_device())


def _get_flash_kernel(
    *, hq: int, sq: int, skv: int, d: int, hkv: int, scaling: float
):
    """Return a cached tilelang flash kernel, or None if tilelang is unavailable.

    The DSL kernel is **shape-generic**: ``B/Hq/Sq/Hkv/Skv`` are dynamic, so a
    single build serves every request. Only ``d`` (head dim, which sizes the
    on-chip tiles) and ``scaling`` (baked as ``sm_scale``) are compile-time, so
    the cache key is just ``(d, scaling)`` — ``hq``/``sq``/``skv``/``hkv`` are
    accepted for call-site symmetry but do not trigger a rebuild.

    ``sq``/``skv`` must still be padded to the kernel's tile sizes (see
    :meth:`TpuAttnBackend._attend_flash`); the DSL kernel does not clamp partial
    tiles, so every tile it loads must be fully in range.
    """
    global _FLASH_DISABLED
    if _FLASH_DISABLED:
        return None
    key = (d, float(scaling))
    kern = _FLASH_KERNELS.get(key)
    if kern is None:
        try:
            import sys

            if "/workspace/tilelang" not in sys.path:
                sys.path.insert(0, "/workspace/tilelang")
            from tilelang.tpu.kernels.attention import flash_attention_gqa

            # Lazy factory call (dynamic-shape idiom): compiles one generic
            # kernel; runtime dims are resolved per call from the input shapes.
            kern = flash_attention_gqa(d=d, sm_scale=float(scaling))
            _FLASH_KERNELS[key] = kern
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[TPU] flash-attention kernel unavailable (%s); "
                "falling back to per-head matmul loop.",
                exc,
            )
            _FLASH_DISABLED = True
            return None
    return kern


class TpuAttnBackend(TorchNativeAttnBackend):
    """Per-head 2D-matmul attention for the Sophgo TPU device."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)

    # ------------------------------------------------------------------
    # Core per-request attention math (replaces SDPA)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_additive_mask(
        *,
        q_len: int,
        kv_len: int,
        causal: bool,
        sliding_window_size: Optional[int],
    ) -> Optional[torch.Tensor]:
        """Build an additive [q_len, kv_len] float mask on CPU, or None.

        Query token ``i`` sits at absolute position ``query_offset + i`` where
        ``query_offset = kv_len - q_len`` (the extend tokens are the tail of the
        sequence; for decode ``q_len == 1``). A key ``j`` is disallowed (masked to
        ``_MASK_NEG``) when it is in the future (``j > pos``) or, for sliding
        window, older than the window (``j < pos - window``).
        """
        has_window = sliding_window_size is not None and sliding_window_size > -1
        if not causal and not has_window:
            return None

        query_offset = kv_len - q_len
        pos = torch.arange(q_len).unsqueeze(1) + query_offset  # [q_len, 1]
        j = torch.arange(kv_len).unsqueeze(0)  # [1, kv_len]

        disallowed = torch.zeros(q_len, kv_len, dtype=torch.bool)
        if causal:
            disallowed |= j > pos
        if has_window:
            disallowed |= j < pos - sliding_window_size

        mask = torch.zeros(q_len, kv_len, dtype=torch.float32)
        mask.masked_fill_(disallowed, _MASK_NEG)
        return mask

    @staticmethod
    def _attend_flash(
        *,
        q: torch.Tensor,  # [H_q, S_q, D]
        k: torch.Tensor,  # [H_kv, S_kv, D]
        v: torch.Tensor,  # [H_kv, S_kv, D]
        out: torch.Tensor,  # [H_q, S_q, D] (written in place)
        scaling: float,
        cpu_mask: Optional[torch.Tensor],  # [S_q, S_kv] additive, or None
    ) -> bool:
        """Try the tilelang PPL flash-attention kernel. Returns True on success
        (``out`` written), False to fall back to the per-head matmul loop.

        The DSL kernel has fixed on-chip tiles (block_M=8 rows of queries,
        block_N=128 keys) and no partial-tile clamping, so ``S_q``/``S_kv`` are
        padded up to those multiples here and sliced back afterwards. Padded
        query rows and padded keys are written as zeros in q/k/v; the mask's
        padded *key columns* are set to ``-30000`` so a real query never puts
        softmax mass on a padded key (whose V is meaningless). Padded *query
        rows* need no masking -- their outputs are sliced off.
        """
        num_q_heads, q_len, d = q.shape
        num_kv_heads, kv_len, _ = k.shape

        block_m, block_n = 8, 128
        q_pad = -(-q_len // block_m) * block_m  # ceil to block_M
        kv_pad = -(-kv_len // block_n) * block_n  # ceil to block_N
        if kv_pad < q_pad:  # the mask builder needs query_offset >= 0
            kv_pad = q_pad

        kernel = _get_flash_kernel(
            hq=num_q_heads,
            sq=q_pad,
            skv=kv_pad,
            d=d,
            hkv=num_kv_heads,
            scaling=scaling,
        )
        if kernel is None:
            return False
        try:
            # [H,S,D] -> [1,H,s_pad,D] fp16 host, zero-padded to the kernel tiles.
            # The dtype cast happens on the host: the device only supports
            # same-dtype copies, and the kernel marshals its arguments through
            # CPU memory regardless.
            def _pad(x, n_heads, s_pad):  # [H,S,D] -> [1,H,s_pad,D]
                padded = torch.zeros(1, n_heads, s_pad, d, dtype=torch.float16)
                padded[0, :, : x.shape[1]] = x.detach().cpu().half()
                return padded

            q_b = _pad(q, num_q_heads, q_pad)
            k_b = _pad(k, num_kv_heads, kv_pad)
            v_b = _pad(v, num_kv_heads, kv_pad)

            if cpu_mask is not None:
                m = cpu_mask.float().clamp_min(_FLASH_MASK_NEG).cpu()
            else:
                # No constraint at all (decode attends every cached key): the
                # real block is all-zeros, and the padded key columns below are
                # masked off.
                m = torch.zeros(q_len, kv_len, dtype=torch.float32)

            if q_pad != q_len or kv_pad != kv_len:
                # Mask row r <-> query row r, column j <-> key j (the kernel
                # bakes in no causal constraint). Padding starts at -30000 so
                # the added key columns are masked out; the padded query rows
                # are overwritten with 0 because the kernel's softmax is taken
                # over a whole query tile -- an all-(-30000) row would make the
                # tile-wide max -30000 and blank the real rows beside it.
                padded = torch.full(
                    (q_pad, kv_pad), _FLASH_MASK_NEG, dtype=torch.float32
                )
                padded[:q_len, :kv_len] = m
                padded[q_len:, :] = 0.0
                m = padded
            mask = m.unsqueeze(0).contiguous()  # [1, S_q, S_kv]

            # Any device-tensor op above (there should be none -- this path
            # marshals through CPU memory) would desync the PPL runtime's
            # current device from its stream. Re-assert it before launching.
            _sync_rt_device()
            res = kernel(q_b, k_b, v_b, mask)  # [1,Hq,q_pad,D] fp16
            # [1,Hq,q_pad,D] -> [Hq,q_len,D]; cast + move on the host.
            resc = res[0, :, :q_len].contiguous()
            out.copy_(resc.to(out.dtype).to(out.device))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[TPU] flash-attention launch failed (%s); "
                "falling back to per-head matmul loop.",
                exc,
            )
            return False

    def _attend_per_req(
        self,
        *,
        q: torch.Tensor,  # [H_q, S_q, D]
        k: torch.Tensor,  # [H_kv, S_kv, D]
        v: torch.Tensor,  # [H_kv, S_kv, D]
        out: torch.Tensor,  # [H_q, S_q, D] (written in place)
        scaling: float,
        causal: bool,
        sliding_window_size: Optional[int],
    ) -> None:
        num_q_heads, q_len, _ = q.shape
        num_kv_heads, kv_len, _ = k.shape
        group_size = num_q_heads // num_kv_heads

        cpu_mask = self._build_additive_mask(
            q_len=q_len,
            kv_len=kv_len,
            causal=causal,
            sliding_window_size=sliding_window_size,
        )

        if envs.SGLANG_TPU_USE_FLASH_ATTN.get() and self._attend_flash(
            q=q, k=k, v=v, out=out, scaling=scaling, cpu_mask=cpu_mask
        ):
            return

        mask = cpu_mask.to(q.device) if cpu_mask is not None else None

        for h in range(num_q_heads):
            kv_h = h // group_size
            q_i = q[h].contiguous()  # [S_q, D]
            k_i = k[kv_h].contiguous()  # [S_kv, D]
            v_i = v[kv_h].contiguous()  # [S_kv, D]

            # bf16 2D matmul; operands must be contiguous or the result is garbage.
            scores = q_i @ k_i.t().contiguous()  # [S_q, S_kv]
            scores = scores.float() * scaling
            if mask is not None:
                scores = scores + mask
            probs = torch.softmax(scores, dim=-1).to(v_i.dtype)
            out[h] = probs.contiguous() @ v_i  # [S_q, D]

    # ------------------------------------------------------------------
    # Overrides of the SDPA extend/decode drivers
    # ------------------------------------------------------------------

    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor] = None,
        scaling=None,
        enable_gqa=False,
        causal=False,
        is_cross_attn=False,
        sliding_window_size: Optional[int] = None,
    ):
        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            extend_seq_len_q = int(extend_seq_lens[seq_idx])
            seq_len_kv = int(seq_lens[seq_idx])
            end_q = start_q + extend_seq_len_q
            if encoder_lens is not None:
                if is_cross_attn:
                    start_kv = 0
                    end_kv = int(encoder_lens[seq_idx])
                else:
                    start_kv = int(encoder_lens[seq_idx])
                    end_kv = start_kv + seq_len_kv
            else:
                start_kv = 0
                end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, start_kv:end_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            per_req_out = torch.empty_like(per_req_query)
            self._attend_per_req(
                q=per_req_query,
                k=per_req_key,
                v=per_req_value,
                out=per_req_out,
                scaling=scaling,
                causal=causal,
                sliding_window_size=sliding_window_size,
            )
            output[start_q:end_q, :, :] = per_req_out.movedim(query.dim() - 2, 0)
            start_q, start_kv = end_q, end_kv
        return output

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor] = None,
        scaling=None,
        enable_gqa=False,
        causal=False,
        is_cross_attn=False,
        sliding_window_size: Optional[int] = None,
    ):
        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            seq_len_q = 1
            seq_len_kv = int(seq_lens[seq_idx])
            end_q = start_q + seq_len_q
            if encoder_lens is not None:
                if is_cross_attn:
                    start_kv = 0
                    end_kv = int(encoder_lens[seq_idx])
                else:
                    start_kv = int(encoder_lens[seq_idx])
                    end_kv = start_kv + seq_len_kv
            else:
                start_kv = 0
                end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, start_kv:end_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            per_req_out = torch.empty_like(per_req_query)
            self._attend_per_req(
                q=per_req_query,
                k=per_req_key,
                v=per_req_value,
                out=per_req_out,
                scaling=scaling,
                # A decode step attends all cached keys; the driver passes
                # causal=False (every key is in the past).
                causal=causal,
                sliding_window_size=sliding_window_size,
            )
            output[start_q:end_q, :, :] = per_req_out.movedim(query.dim() - 2, 0)
            start_q, start_kv = end_q, end_kv

        return output

    def support_triton(self):
        return False
