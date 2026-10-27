"""Fused post-attention MLP for the TPU path.

One launch replaces three, and deletes the 100 MB bf16 intermediate they hand
each other::

    gate_up, _ = self.gate_up_proj(x)     # [T, 2I]  -> DRAM
    mid        = self.act_fn(gate_up)     # [T, I]   <- DRAM, -> DRAM
    x, _       = self.down_proj(mid)      # [T, H]   <- DRAM

The kernel (``tilelang.tpu.kernels.fused_mlp``) keeps the product in registers
and feeds it straight into the down projection, so ``[T, I]`` never reaches
DRAM.  That traffic -- not the launch count -- is what the fusion buys: at
T=8192 the two elementwise-bound rows of the per-layer profile move ~200 MB
between them.

``x`` is the *normalized* hidden state: ``LayerCommunicator.prepare_mlp`` has
already applied ``post_attention_layernorm`` by the time the layer calls this,
so the kernel needs no norm of its own.

The path is switched by ``SGLANG_TPU_USE_FUSED_MLP`` and gated only on geometry
at construction time: an unquantized bf16 MLP (the kernel is built for bf16
weights), tensor parallelism of one (the kernel would have to reproduce
``down_proj``'s all-reduce), and ``mlp_geometry_ok``.  There is deliberately no
size gate -- the caller decides.

Size does matter for the win, though, and the flag is a manual switch precisely
because of it: measured against the torch chain on SG2260E (Qwen3-0.6B,
``test_fused_mlp.py --sweep``), the fusion is 1.70x at T=8192, 1.61x at 4096,
1.21x at 2048, 1.40x at 1024, then loses -- 0.83x at T=512 and 0.79x at T=256.
Below a few hundred rows the MLP is a pure weight stream that the torch chain
spreads over all four cores and one row tile cannot.
"""

from __future__ import annotations

import torch

#: Compiled kernels, keyed by the full geometry + tile + dtype.  The build is
#: ~1 minute, so the cache is process-wide and ``__call__`` is the only place
#: that pays for it.
_KERNELS: dict[tuple, object] = {}

_BLOCK_I = 256

#: Reduction tile over ``hidden``, largest first.  ``block_K == hidden`` is the
#: fastest measured shape: it moves the same bytes as any smaller tile but
#: issues one DMA per K sweep instead of ``hidden / block_K`` of them
#: (T=8192: 4.41 ms at 1024 vs 5.00 ms at 256).  The ladder exists only for
#: hidden sizes that are not multiples of the largest entry.
_BLOCK_K_LADDER = (1024, 512, 256)

#: Row tile in tokens, taken as the largest one that still gives every core a
#: tile.  ``block_M`` sets the weight traffic -- a row tile streams the whole
#: 18.9 MB gate/up/down set -- so it is chosen as large as the batch allows,
#: never larger.  Values above the ``T / cores`` line are pure idle core.
_BLOCK_M_LADDER = (1024, 512, 256, 128, 64)
_NUM_CORES = 4


def mlp_geometry_ok(
    *,
    hidden_size: int,
    intermediate_size: int,
    block_i: int = _BLOCK_I,
) -> bool:
    """Whether the kernel can be built for this MLP geometry at all."""
    return (
        hidden_size % _BLOCK_K_LADDER[-1] == 0
        and intermediate_size % block_i == 0
    )


def _pick_block_k(hidden_size: int) -> int:
    """The largest reduction tile the hidden size allows."""
    for block_k in _BLOCK_K_LADDER:
        if hidden_size % block_k == 0:
            return block_k
    return _BLOCK_K_LADDER[-1]


def _pick_block_m(num_tokens: int) -> int:
    """The largest tile on the ladder that puts a tile on every core."""
    for block_m in _BLOCK_M_LADDER:
        if num_tokens >= block_m * _NUM_CORES:
            return block_m
    return _BLOCK_M_LADDER[-1]


def _get_kernel(
    *,
    hidden_size: int,
    intermediate_size: int,
    block_m: int,
    torch_dtype: torch.dtype,
):
    block_k = _pick_block_k(hidden_size)
    key = (
        hidden_size,
        intermediate_size,
        block_m,
        _BLOCK_I,
        block_k,
        str(torch_dtype),
    )
    kernel = _KERNELS.get(key)
    if kernel is None:
        from tilelang.tpu.kernels.fused_mlp import fused_mlp

        kernel = fused_mlp(
            hidden=hidden_size,
            intermediate=intermediate_size,
            block_M=block_m,
            block_I=_BLOCK_I,
            block_K=block_k,
            dtype=str(torch_dtype).replace("torch.", ""),
        )
        _KERNELS[key] = kernel
    return kernel


class TpuFusedMlp:
    """gate_up_proj -> silu_mul -> down_proj, for one layer.

    Holds references to the two projections' weights (so a reload is picked up
    without a rebuild) and one padded input buffer per token count.
    """

    def __init__(
        self,
        *,
        gate_up_proj,
        down_proj,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        self._gate_up_proj = gate_up_proj
        self._down_proj = down_proj
        self._hidden_size = hidden_size
        self._intermediate_size = intermediate_size
        self._x_pad: torch.Tensor | None = None
        self._y_pad: torch.Tensor | None = None
        self._buf_key: tuple | None = None

    def _padded_buffers(
        self, *, padded: int, dtype: torch.dtype, device
    ) -> None:
        """``[padded, H]`` input and output tiles, ``padded`` a tile multiple.

        The kernel's row tile is baked, so a batch that is not a multiple of
        ``block_M`` must be padded rather than clamped.  Rows past the batch are
        never rewritten once the buffers exist: they hold whatever the last
        larger batch left there, which is harmless because a row's output
        depends only on that row and the caller slices the result back to the
        token count.
        """
        key = (padded, self._hidden_size, dtype, device)
        if self._buf_key == key:
            return
        self._x_pad = torch.empty(
            (padded, self._hidden_size), dtype=dtype, device=device
        )
        self._y_pad = torch.empty(
            (padded, self._hidden_size), dtype=dtype, device=device
        )
        self._buf_key = key

    def __call__(self, *, x: torch.Tensor) -> torch.Tensor:
        """Return the MLP output for the already-normalized ``x``."""
        num_tokens, hidden_size = x.shape
        block_m = _pick_block_m(num_tokens)
        padded = -(-num_tokens // block_m) * block_m

        kernel = _get_kernel(
            hidden_size=hidden_size,
            intermediate_size=self._intermediate_size,
            block_m=block_m,
            torch_dtype=x.dtype,
        )
        self._padded_buffers(padded=padded, dtype=x.dtype, device=x.device)

        # The kernel addresses X as a flat row-major matrix off its data
        # pointer, so the row pitch must be exactly ``hidden_size``.
        #
        # Copy only when the batch is not already a tile multiple: the copy is a
        # full read+write of ``[T, H]`` (~0.7 ms at T=8192), which is the same
        # order as the whole saving the fusion buys, and prefill batches are
        # tile-aligned by construction -- `_pick_block_m` returns a power of two
        # and the token counts that matter are multiples of it.
        if padded == num_tokens and x.is_contiguous():
            x_in = x
        else:
            self._x_pad[:num_tokens] = x
            x_in = self._x_pad
        kernel(
            x_in,
            self._gate_up_proj.weight,
            self._down_proj.weight,
            self._y_pad,
        )
        return self._y_pad[:num_tokens]
