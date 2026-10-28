"""Post-attention RMSNorm folded into the fused MLP (TPU).

Milestone C of :mod:`fused_mlp`: the same one-launch MLP with the decoder layer's
``post_attention_layernorm`` inside it::

    h    = hidden + residual                 # the *layer* does this, not the kernel
    x    = post_attention_layernorm(h)       # <- kernel
    gu   = gate_up_proj(x)                   # <- kernel
    mid  = silu(gu[..., :I]) * gu[..., I:]   # <- kernel
    y    = down_proj(mid)                    # <- kernel

The kernel takes ``h`` un-normalized, because that sum is also the value the
layer carries forward as its next residual -- so folding the norm in removes the
``prepare_mlp`` boundary instead of just moving it.  The layer is therefore
responsible for reproducing ``prepare_mlp``'s residual hand-off; see
``Qwen3DecoderLayer.forward``.  Where ``h`` comes from is the layer's business:
``_fused_ln_residual`` forms it today, and ``TpuOprojRes`` forms it inside the
``o_proj`` kernel when that op is live -- both hand over the same raw sum, and
this op cannot tell which.

The norm's ``gamma`` is folded into the gate/up weight here, once per layer:
``gate = rstd * (x . (gamma * Wgu)^T)``, so ``gamma`` is a per-column scale of
``Wgu``.  The kernel must never see an elementwise op between a weight tile's
load and the GEMM that consumes it (it corrupts the accumulator -- see the
``fused_qkv_rope_ln`` docstring), so the fold happens on the driver.  ``Wd`` is
``[H, I]``: ``H`` is its *output* axis, so it is not folded.
"""

from __future__ import annotations

import torch

from sglang.srt.hardware_backend.tpu.fused_mlp import (
    _BLOCK_I,
    _pick_block_k,
    TpuFusedMlp,
)

#: Compiled kernels, keyed like the plain op's cache plus the norm epsilon.
_KERNELS: dict[tuple, object] = {}


def _get_kernel(
    *,
    hidden_size: int,
    intermediate_size: int,
    block_m: int,
    ln_eps: float,
    torch_dtype: torch.dtype,
):
    block_k = _pick_block_k(hidden_size)
    key = (
        hidden_size,
        intermediate_size,
        block_m,
        _BLOCK_I,
        block_k,
        float(ln_eps),
        str(torch_dtype),
    )
    kernel = _KERNELS.get(key)
    if kernel is None:
        from tilelang.tpu.kernels.fused_mlp_ln import fused_mlp_ln

        kernel = fused_mlp_ln(
            hidden=hidden_size,
            intermediate=intermediate_size,
            block_M=block_m,
            block_I=_BLOCK_I,
            block_K=block_k,
            ln_eps=ln_eps,
            dtype=str(torch_dtype).replace("torch.", ""),
        )
        _KERNELS[key] = kernel
    return kernel


class TpuFusedMlpLn(TpuFusedMlp):
    """``post_attention_layernorm`` -> gate_up -> silu_mul -> down, one launch.

    Everything the plain wrapper does applies unchanged; the two differences are
    the norm's epsilon (baked into the kernel) and the gate/up weight, which is
    read through ``post_attention_layernorm.weight``::

        w_folded = (gate_up_proj.weight.float() * post_attention_layernorm.weight).to(dtype)

    That fold is one ``[2I, H]`` elementwise pass per layer, cached until either
    weight's storage or in-place version changes, so a launch pays nothing for
    it.

    ``__call__`` is the base class's, unchanged: ``x`` is the layer's raw
    ``hidden + residual`` (the kernel normalizes it), not a normalized hidden.
    """

    _kernel_trace_name = "fused_mlp_ln"

    def __init__(
        self,
        *,
        gate_up_proj,
        down_proj,
        post_attention_layernorm,
        hidden_size: int,
        intermediate_size: int,
        ln_eps: float,
    ) -> None:
        super().__init__(
            gate_up_proj=gate_up_proj,
            down_proj=down_proj,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        self._post_attention_layernorm = post_attention_layernorm
        self._ln_eps = float(ln_eps)
        self._w_folded: torch.Tensor | None = None
        self._w_folded_key: tuple | None = None

    def _make_kernel(self, *, hidden_size: int, block_m: int, torch_dtype):
        return _get_kernel(
            hidden_size=hidden_size,
            intermediate_size=self._intermediate_size,
            block_m=block_m,
            ln_eps=self._ln_eps,
            torch_dtype=torch_dtype,
        )

    def _projection_weight(self) -> torch.Tensor:
        """``gamma_post_attn * Wgu``, folded once per weight version."""
        w, gin = self._gate_up_proj.weight, self._post_attention_layernorm.weight
        key = (w.data_ptr(), w._version, gin.data_ptr(), gin._version)
        if self._w_folded is None or self._w_folded_key != key:
            self._w_folded = (w.float() * gin.reshape(1, -1).float()).to(w.dtype)
            self._w_folded_key = key
        return self._w_folded
