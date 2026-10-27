"""Input-RMSNorm folded into the fused pre-attention chain (TPU).

Milestone B of :mod:`fused_qkv_rope`: the same one-launch chain with the decoder
layer's ``input_layernorm`` inside it::

    h = hidden + residual                    # the *layer* does this, not the kernel
    normed = input_layernorm(h)              # <- kernel
    qkv, _ = self.qkv_proj(normed)           # <- kernel
    q, k   = apply_qk_norm(q, k, ...)        # <- kernel
    q, k   = self.rotary_emb(...)            # <- kernel
    k_cache[loc], v_cache[loc] = k, v        # <- kernel

The kernel takes ``h`` **un-normalized**, because that sum is also the value the
layer carries forward as its next residual -- so folding the norm in removes the
``prepare_attn`` boundary entirely instead of just moving it.  The layer is
therefore responsible for reproducing ``prepare_attn``'s residual hand-off; see
``Qwen3DecoderLayer.forward``.

The input ``gamma`` is folded into the projection weight here, once per layer:
``q = rstd * (x . (gamma_in * W)^T)``, so ``gamma_in`` is a per-column scale of
``W``.  The kernel must never see an elementwise op between a weight tile's load
and the GEMM that consumes it (it corrupts the accumulator -- see the kernel's
docstring), so the fold happens on the driver and the kernel's k-loop is
identical to the projection-only kernel's.
"""

from __future__ import annotations

import torch

from sglang.srt.hardware_backend.tpu.fused_qkv_rope import (
    _BLOCK_K,
    TpuFusedQkvRope,
)

#: Compiled kernels, keyed like the M-A cache plus the input-norm epsilon.
_KERNELS: dict[tuple, object] = {}


def _get_kernel(
    *,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_m: int,
    eps: float,
    ln_eps: float,
    torch_dtype: torch.dtype,
):
    key = (
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        block_m,
        _BLOCK_K,
        float(eps),
        float(ln_eps),
        str(torch_dtype),
    )
    kernel = _KERNELS.get(key)
    if kernel is None:
        from tilelang.tpu.kernels.fused_qkv_rope_ln import fused_qkv_rope_ln

        kernel = fused_qkv_rope_ln(
            hidden=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_M=block_m,
            block_K=_BLOCK_K,
            eps=eps,
            ln_eps=ln_eps,
            dtype=str(torch_dtype).replace("torch.", ""),
        )
        _KERNELS[key] = kernel
    return kernel


class TpuFusedQkvLn(TpuFusedQkvRope):
    """``input_layernorm`` -> qkv -> qk-norm -> RoPE -> paged-cache write.

    Everything the M-A wrapper does applies unchanged; the two differences are
    the input norm's epsilon (baked into the kernel) and the projection weight,
    which is read through ``input_layernorm.weight``::

        w_folded = (qkv_proj.weight.float() * input_layernorm.weight.float()).to(dtype)

    That fold is one ``[Ntot, H]`` elementwise pass per layer, cached until
    either weight's storage or in-place version changes, so a launch pays
    nothing for it.

    ``__call__`` is the base class's, unchanged: ``hidden_states`` is the layer's
    raw ``hidden + residual`` (the kernel normalizes it), not a normalized one.
    """

    _kernel_trace_name = "fused_qkv_rope_ln"

    def __init__(
        self,
        *,
        qkv_proj,
        input_layernorm,
        q_norm,
        k_norm,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        ln_eps: float,
    ) -> None:
        super().__init__(
            qkv_proj=qkv_proj,
            q_norm=q_norm,
            k_norm=k_norm,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self._input_layernorm = input_layernorm
        self._ln_eps = float(ln_eps)
        self._w_folded: torch.Tensor | None = None
        self._w_folded_key: tuple | None = None

    def _make_kernel(self, *, hidden_size: int, block_m: int, eps: float, torch_dtype):
        return _get_kernel(
            hidden_size=hidden_size,
            num_heads=self._num_heads,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            block_m=block_m,
            eps=eps,
            ln_eps=self._ln_eps,
            torch_dtype=torch_dtype,
        )

    def _projection_weight(self) -> torch.Tensor:
        """``gamma_in * W_qkv``, folded once per weight version."""
        w, gin = self._qkv_proj.weight, self._input_layernorm.weight
        key = (w.data_ptr(), w._version, gin.data_ptr(), gin._version)
        if self._w_folded is None or self._w_folded_key != key:
            self._w_folded = (w.float() * gin.reshape(1, -1).float()).to(w.dtype)
            self._w_folded_key = key
        return self._w_folded
