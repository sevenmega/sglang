"""Fused ``o_proj`` + residual add for the TPU path.

One launch replaces the aten GEMM and the four-op residual dance that follows
it::

    y, _ = self.o_proj(attn_output)              # [T, H] -> DRAM
    h    = (y.float() + residual.float()).to(bf16)

The kernel (``tilelang.tpu.kernels.oproj_res``) seeds its fp32 accumulator with
the residual, so ``y`` never reaches DRAM and the sum leaves in one narrowing.
Measured on SG2260E at T=8192 (``profile_one_layer.py``): the GEMM row 2.07 ms,
the fused row 1.98 ms, and the dance 1.65 ms on its own -- so the boundary is
~3.7 ms and the fusion ~2.0 ms.  The dance is four eager aten kernels (three
full ``[T, H]`` passes plus two fp32 temporaries) and used to appear in no row
at all, because the layer-e2e call passed ``residual=None`` and it returned
early.

``h`` is the *raw* post-attention sum, not a normalized one: it is what
``TpuFusedMlpLn`` consumes on chip and, at the same time, the layer's carried
residual.  That contract is why this op requires the norm-fused MLP -- with any
other MLP path ``prepare_mlp`` would add the residual a second time -- and the
caller (``Qwen3DecoderLayer.forward``) is responsible for the pairing.  Layer 0
has no residual and keeps the plain ``o_proj``.

Gated at construction time on geometry: ``o_proj.bias is None`` (a bias would
change the kernel's ABI), tensor parallelism of one (the kernel does not
all-reduce), and ``oproj_geometry_ok``.  Nothing else gates it -- with the flag
set, every batch fuses, decode included, and the caller owns that choice.

That matters because the fusion *loses* below about 4096 tokens: the kernel's
floor is a launch plus one 4 MB weight stream, which a small batch cannot
amortize, while the aten kernels it replaces are individually cheap.  Per layer,
fused against ``mm`` + dance: T=64 0.41 ms against 0.22, T=1024 1.10 against
0.84, T=2048 0.80 against 0.73, T=4096 0.95 against 1.73, T=8192 1.94 against
3.34.  The row tile is padded to ``_BLOCK_M`` regardless of batch, so a decode
step costs the same ~0.4 ms per layer a T=64 batch does.  Leave the flag off
unless the deployment is prefill-heavy (or prefill/decode disaggregated); the
engine has no per-batch switch, so "on" taxes every decode step.
"""

from __future__ import annotations

import torch

from sglang.srt.hardware_backend.tpu.trace import trace_tpu_kernel

#: Compiled kernels, keyed by the full geometry + tile + dtype.  The build is
#: ~1 minute, so the cache is process-wide and ``__call__`` is the only place
#: that pays for it.
_KERNELS: dict[tuple, object] = {}

#: Reduction tile over ``input_size`` (``Hq*D``), largest first.  As in the MLP,
#: ``block_K == input_size`` is the fastest shape: same bytes, one DMA per K
#: sweep instead of ``input_size / block_K``.
_BLOCK_K_LADDER = (1024, 512, 256)

#: Row tile in tokens, baked into the one kernel this op ever builds.  Larger is
#: better at equal core occupancy -- a tile streams the whole 4 MB weight, so at
#: T=4096 ``1024`` measures 0.95 ms against 1.05 for ``512``, and at T=8192 1.94
#: against 2.14 -- and the one size where the smaller tile wins (T=2048: 0.70
#: against 0.80, because 1024 leaves two cores idle) is not worth a second tile.
#: Fixed rather than adapted per batch: a second tile in one process means a
#: second kernel config, and this runtime has been seen to hang initializing
#: one (see ``test_oproj_res.py``).  Padding covers any batch instead.
_BLOCK_M = 1024

#: Columns are aligned to this in the emitter's local-memory accounting, so a
#: geometry whose widths are not multiples of it would neither pad nor tile
#: cleanly.
_ALIGN = 16


def oproj_geometry_ok(*, hidden_size: int, input_size: int) -> bool:
    """Whether the kernel can be built for this ``o_proj`` geometry at all."""
    return (
        hidden_size % _ALIGN == 0
        and input_size % _BLOCK_K_LADDER[-1] == 0
    )


def _pick_block_k(input_size: int) -> int:
    """The largest reduction tile the projection's input width allows."""
    for block_k in _BLOCK_K_LADDER:
        if input_size % block_k == 0:
            return block_k
    return _BLOCK_K_LADDER[-1]


def _get_kernel(
    *,
    hidden_size: int,
    input_size: int,
    block_m: int,
    torch_dtype: torch.dtype,
):
    block_k = _pick_block_k(input_size)
    key = (hidden_size, input_size, block_m, block_k, str(torch_dtype))
    kernel = _KERNELS.get(key)
    if kernel is None:
        from tilelang.tpu.kernels.oproj_res import oproj_res

        kernel = oproj_res(
            hidden=hidden_size,
            input_size=input_size,
            block_M=block_m,
            block_K=block_k,
            dtype=str(torch_dtype).replace("torch.", ""),
        )
        _KERNELS[key] = kernel
    return kernel


class TpuOprojRes:
    """``o_proj`` with the residual folded into its accumulator, for one layer.

    Holds a reference to the projection (so a weight reload is picked up without
    a rebuild) and one padded buffer set per token count.
    """

    _kernel_trace_name = "oproj_res"

    def __init__(self, *, o_proj, hidden_size: int, input_size: int) -> None:
        self._o_proj = o_proj
        self._hidden_size = hidden_size
        self._input_size = input_size
        self._a_pad: torch.Tensor | None = None
        self._r_pad: torch.Tensor | None = None
        self._y_pad: torch.Tensor | None = None
        self._buf_key: tuple | None = None

    def _make_kernel(self, *, input_size: int, block_m: int, torch_dtype):
        """The compiled kernel for this layer's geometry and tile."""
        return _get_kernel(
            hidden_size=self._hidden_size,
            input_size=input_size,
            block_m=block_m,
            torch_dtype=torch_dtype,
        )

    def _projection_weight(self) -> torch.Tensor:
        """The ``[H, Hq*D]`` weight the kernel contracts against."""
        return self._o_proj.weight

    def _padded_buffers(
        self, *, padded: int, dtype: torch.dtype, device
    ) -> None:
        """``[padded, K]`` / ``[padded, H]`` tiles, ``padded`` a tile multiple.

        As in the MLP: the row tile is baked, so an unaligned batch is padded
        rather than clamped, and rows past the batch are never rewritten -- a
        row's output depends only on that row and the caller slices back.
        """
        key = (padded, self._input_size, self._hidden_size, dtype, device)
        if self._buf_key == key:
            return
        self._a_pad = torch.empty(
            (padded, self._input_size), dtype=dtype, device=device
        )
        self._r_pad = torch.empty(
            (padded, self._hidden_size), dtype=dtype, device=device
        )
        self._y_pad = torch.empty(
            (padded, self._hidden_size), dtype=dtype, device=device
        )
        self._buf_key = key

    def __call__(
        self, *, attn_output: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        """Return ``attn_output @ W_o^T + residual``, the layer's new hidden state."""
        num_tokens, input_size = attn_output.shape
        block_m = _BLOCK_M
        padded = -(-num_tokens // block_m) * block_m

        kernel = self._make_kernel(
            input_size=input_size, block_m=block_m, torch_dtype=attn_output.dtype
        )
        self._padded_buffers(
            padded=padded, dtype=attn_output.dtype, device=attn_output.device
        )

        # Both operands are addressed flat off their data pointer, so the row
        # pitch must be exact.  Copy only when the batch is not already a tile
        # multiple (prefill batches are, by construction) -- see ``TpuFusedMlp``.
        if padded == num_tokens and attn_output.is_contiguous():
            attn_in = attn_output
        else:
            self._a_pad[:num_tokens] = attn_output
            attn_in = self._a_pad
        if padded == num_tokens and residual.is_contiguous():
            residual_in = residual
        else:
            self._r_pad[:num_tokens] = residual
            residual_in = self._r_pad

        # The kernel writes only its own buffer.  A residual pointing at it
        # would be a read-write hazard: rows one tile wrote would seed another.
        assert residual_in.data_ptr() != self._y_pad.data_ptr()

        with trace_tpu_kernel(
            self._kernel_trace_name,
            tokens=num_tokens,
            padded=padded,
            block_m=block_m,
        ):
            kernel(
                attn_in,
                self._projection_weight(),
                residual_in,
                self._y_pad,
            )
        return self._y_pad[:num_tokens]
