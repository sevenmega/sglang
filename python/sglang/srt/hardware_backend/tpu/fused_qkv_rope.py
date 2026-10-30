"""Fused pre-attention chain for the TPU attention path.

One launch replaces what a Qwen3 decoder layer spends ~15 launches on today::

    qkv, _ = self.qkv_proj(hidden)           # F.linear
    q, k, v = qkv.split(...)                 # strided views
    q, k = apply_qk_norm(q, k, ...)          # 2 reshape copies + 2 RMSNorms
    q, k = self.rotary_emb(positions, q, k)  # index_select + chunk/mul/add/cat
    k_cache[out_cache_loc] = k               # 2 strided scatters
    v_cache[out_cache_loc] = v

The kernel (``tilelang.tpu.kernels.fused_qkv_rope``) writes ``Q`` for the paged
attention kernel and writes ``K``/``V`` straight into the paged cache, so the
driver never materializes ``k``/``v`` at all -- the caller must pass
``save_kv_cache=False`` (and may pass ``k=v=None``) to the attention backend.

Degenerate layouts this path cannot serve keep the torch chain: a non-NeoX or
partially-rotated head, a non-bf16 activation, a cross-attention or
sliding-window layer, and any head geometry the kernel was not built for.  The
model checks those once at construction (``_enable_tpu_fused_qkv_rope``), so
``__call__`` never re-tests them.
"""

from __future__ import annotations

import msgspec
import torch

from sglang.srt.hardware_backend.tpu.trace import trace_tpu_kernel
from sglang.srt.model_executor.forward_context import get_attn_backend

#: Compiled kernels, keyed by the full geometry + tile + dtype.  The build is
#: ~1 minute, so the cache is process-wide and the wrap's ``__call__`` is the
#: only place that pays for it.
_KERNELS: dict[tuple, object] = {}


class _ForwardRows(msgspec.Struct):
    """The cos/sin rows and KV write table for one forward, shared by its layers.

    Both depend on the batch alone -- ``positions`` and the paged slots -- so the
    first layer of a forward builds them and the rest read the same tensors.
    The source tensors are held so their identities stay unique for the life of
    the entry (the cache keys on ``id``).
    """

    positions: torch.Tensor
    cos_sin_cache: torch.Tensor
    out_cache_loc: torch.Tensor
    cos_rows: torch.Tensor
    kv_row: torch.Tensor

#: Query-row tile in tokens, largest first.  ``block_M`` is the GEMM's M and the
#: kernel's dominant knob: at T=8192 it is 15.05 ms at 64 vs 5.22 ms at 4096,
#: because the tile sets both the fmm2 shape and how many m-tile sweeps the job
#: runs.  It is not free to grow, though -- rows past the batch are padded work,
#: so past the token count a wider tile loses badly (T=256: 0.475 ms at 256 vs
#: 1.691 at 1024).  ``_pick_block_m`` trades the two against each other.
#:
#: Those numbers are the norm-fused kernel's.  The projection-only kernel was
#: spot-checked on the same points and picks the same way -- 12.60 ms at 64 vs
#: 4.26 ms at 4096 at T=8192, and 1.01 ms at 256 vs 1.57 ms at 1024 at T=256
#: (it is uniformly faster: it skips the input-norm pass) -- so one table
#: serves both.
_BLOCK_M_LADDER = (4096, 2048, 1024, 512, 256, 128, 64)
_BLOCK_K = 256

#: Measured cost of one padded token row, in microseconds, per tile width.  The
#: 64-4096 entries come from the T=8192 sweep, 128/256 from T=1024 and T=256.
#: Only the ratios matter -- the pick multiplies them by the padded row count.
#: 64 is the narrowest tile on purpose: at T=1/17/32 it measured 0.455-0.469 ms,
#: the same as a 16-row tile (both are launch-bound), so 16 would only add a
#: kernel build to every serving process for nothing.
_ROW_COST_US = {
    64: 1.84,
    128: 1.32,
    256: 0.90,
    512: 0.75,
    1024: 0.68,
    2048: 0.65,
    4096: 0.64,
}


def _pick_block_m(num_tokens: int) -> int:
    """The tile minimising ``padded rows * measured per-row cost``.

    Minimising padding alone is not enough -- it picks a 16-row tile for a
    1000-token batch -- and maximising the tile alone over-pads (a 4096 tile
    rounds 5000 tokens up to 8192, which costs more than the wider GEMM saves).
    """
    return min(
        _BLOCK_M_LADDER,
        key=lambda block_m: (
            -(-num_tokens // block_m) * block_m * _ROW_COST_US[block_m]
        ),
    )


def kernel_geometry_ok(
    *,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_k: int = _BLOCK_K,
) -> bool:
    """Whether the kernel supports this head geometry at all."""
    return (
        num_kv_heads > 0
        and num_heads % num_kv_heads == 0
        and head_dim % 2 == 0
        and hidden_size % block_k == 0
    )


def _get_kernel(
    *,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_m: int,
    eps: float,
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
        str(torch_dtype),
    )
    kernel = _KERNELS.get(key)
    if kernel is None:
        from tilelang.tpu.kernels.fused_qkv_rope import fused_qkv_rope

        kernel = fused_qkv_rope(
            hidden=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_M=block_m,
            block_K=_BLOCK_K,
            eps=eps,
            dtype=str(torch_dtype).replace("torch.", ""),
        )
        _KERNELS[key] = kernel
    return kernel


class TpuFusedQkvRope:
    """qkv_proj -> qk-norm -> RoPE -> paged-cache write, for one layer.

    Holds the layer's norm weights fused into the fp32 ``[2, head_dim]`` table
    the kernel wants, plus its qkv weight.  Both are read through references to
    the owning modules, so a weight reload is picked up: the fused table is
    rebuilt whenever either norm weight's storage or in-place version changes.
    """

    #: Name the launch tracer prints; subclasses that add work to the same
    #: kernel override it so the trace names what actually ran.
    _kernel_trace_name = "fused_qkv_rope"

    def __init__(
        self,
        *,
        qkv_proj,
        q_norm,
        k_norm,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        self._qkv_proj = qkv_proj
        self._q_norm = q_norm
        self._k_norm = k_norm
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._half = head_dim // 2
        self._gamma: torch.Tensor | None = None
        self._gamma_key: tuple | None = None
        # Everything else the kernel reads is either fixed for the layer's
        # lifetime or a view of storage that never moves, and rebuilding it per
        # call is what made the fused path slower than the torch chain it
        # replaces: ``get_kv_buffer`` alone measured ~4 ms of device time per
        # layer, and each fresh ``torch.zeros``/``cat`` is another allocation.
        # All three caches are keyed on storage identity, so a weight reload or
        # a re-allocated pool rebuilds them.
        self._cache_key: tuple | None = None
        self._k_flat: torch.Tensor | None = None
        self._v_flat: torch.Tensor | None = None
        self._buf_key: tuple | None = None
        self._x_pad: torch.Tensor | None = None

    # -- per-layer weight preparation ------------------------------------- #

    def _gamma_table(self) -> torch.Tensor:
        """``[2, head_dim]`` fp32: q_norm's weight above k_norm's.

        The model's RMSNorm multiplies a bf16 weight into an fp32 activation, so
        widening the bf16 value is exact and the fused table is bit-identical.
        """
        qw, kw = self._q_norm.weight, self._k_norm.weight
        key = (qw.data_ptr(), qw._version, kw.data_ptr(), kw._version)
        if self._gamma is None or self._gamma_key != key:
            self._gamma = torch.cat([qw, kw], dim=0).float().view(2, -1)
            self._gamma_key = key
        return self._gamma

    # -- kernel selection -------------------------------------------------- #

    def _make_kernel(self, *, hidden_size: int, block_m: int, eps: float, torch_dtype):
        """The compiled kernel for this layer's geometry."""
        return _get_kernel(
            hidden_size=hidden_size,
            num_heads=self._num_heads,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            block_m=block_m,
            eps=eps,
            torch_dtype=torch_dtype,
        )

    def _projection_weight(self) -> torch.Tensor:
        """The ``[Ntot, H]`` weight the kernel contracts against."""
        return self._qkv_proj.weight

    # -- per-launch buffers ------------------------------------------------ #

    def _cache_views(self, k_cache, v_cache):
        """The paged cache as flat ``[rows, head_dim // 2]`` half-rows.

        The NHD cache's row ``(slot, h, d)`` is half-row
        ``slot * 2 * Hkv + 2 * h + d // 64``; the kernel only needs the base
        pointer and the half-row pitch, so the view is built once.
        """
        key = (k_cache.data_ptr(), v_cache.data_ptr(), k_cache.shape[0])
        if self._cache_key != key:
            self._k_flat = k_cache.view(-1, self._half)
            self._v_flat = v_cache.view(-1, self._half)
            self._cache_key = key
            self._buf_key = None  # the spare-slot index depends on the pool
        return self._k_flat, self._v_flat

    def _padded_buffers(
        self, *, padded: int, hidden_size: int, dtype: torch.dtype, device
    ):
        """Cached ``X`` tile of ``padded`` rows.

        Rows past the real token count are never rewritten once the buffer
        exists: they hold whatever the last call with a larger token count left
        there, which is safe because a padded row's ``Q`` is sliced off and its
        ``KV`` goes to the pool's reserved spare slot.
        """
        key = (padded, hidden_size, dtype, device)
        if self._buf_key == key:
            return
        self._x_pad = torch.zeros((padded, hidden_size), dtype=dtype, device=device)
        self._buf_key = key

    def _build_cos_rows(self, *, positions, cos_sin_cache, num_tokens, padded, device):
        """``[padded, head_dim]`` fp32 cos|sin rows for this batch's positions.

        The cache itself is bf16 once the assembly path has matched the
        activation dtype, so the copy widens it -- a bf16->fp32 widen has no
        on-chip instruction, but this is a plain elementwise op on the driver
        side, not in the kernel.
        """
        cos_rows = torch.zeros(
            (padded, self._head_dim), dtype=torch.float32, device=device
        )
        cos_rows[:num_tokens] = cos_sin_cache.index_select(0, positions.flatten())
        return cos_rows

    def _build_kv_row(self, *, out_cache_loc, num_tokens, padded, device):
        """``[2*Hkv, padded]`` int32: each KV head's cache row per token.

        Columns past the batch address the pool's spare slot, or the kernel's
        padded rows would write into live cache rows; that is why the whole
        table is seeded with the spare slot before the real columns land.
        ``out_cache_loc`` is int64 and the kernel reads this table as 32-bit
        integers, so narrow it here -- an int64 table would be read at half its
        real stride and yield interleaved garbage rows.
        """
        n_kv = self._num_kv_heads
        kv_row = torch.zeros((2 * n_kv, padded), dtype=torch.int32, device=device)
        rows = torch.arange(2 * n_kv, dtype=torch.int32, device=device)
        # ``_cache_views`` runs first, so the pool's row count is known.
        pad_slot = self._k_flat.shape[0] // (2 * n_kv) - 1
        kv_row[:, :] = rows.view(-1, 1) + pad_slot * (2 * n_kv)
        slots = out_cache_loc.view(1, -1).to(torch.int32)
        kv_row[:, :num_tokens] = slots * (2 * n_kv) + rows.view(-1, 1)
        return kv_row

    def _forward_rows(
        self, *, positions, cos_sin_cache, out_cache_loc, num_tokens, padded, device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The batch's cos|sin rows and KV write table, built once per forward.

        Neither depends on the layer, so every layer of a forward shares one
        copy.  Rebuilding them per layer was ten launches a layer: the
        40960-row cos-table gather, its widen into the fp32 tile, and the table
        arithmetic behind ``kv_row``.
        """
        key = (
            "qkv_rows",
            num_tokens,
            padded,
            id(positions),
            id(cos_sin_cache),
            id(out_cache_loc),
            str(device),
        )

        def build() -> _ForwardRows:
            return _ForwardRows(
                positions=positions,
                cos_sin_cache=cos_sin_cache,
                out_cache_loc=out_cache_loc,
                cos_rows=self._build_cos_rows(
                    positions=positions,
                    cos_sin_cache=cos_sin_cache,
                    num_tokens=num_tokens,
                    padded=padded,
                    device=device,
                ),
                kv_row=self._build_kv_row(
                    out_cache_loc=out_cache_loc,
                    num_tokens=num_tokens,
                    padded=padded,
                    device=device,
                ),
            )

        rows = get_attn_backend().forward_glue(key, build)
        return rows.cos_rows, rows.kv_row

    # -- the launch -------------------------------------------------------- #

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        out_cache_loc: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """Return ``q`` as the kernel's packed ``[padded, num_heads * head_dim]``.

        ``padded`` is ``T`` rounded up to the tile, and only the leading ``T``
        rows are the batch's -- see the return statement for why the padding is
        handed back rather than sliced off here.

        ``K``/``V`` are already in ``k_cache``/``v_cache`` when this returns;
        rows past ``T`` (the tile padding) write to the pool's spare slot only.
        """
        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        block_m = _pick_block_m(num_tokens)
        padded = -(-num_tokens // block_m) * block_m

        kernel = self._make_kernel(
            hidden_size=hidden_size,
            block_m=block_m,
            eps=eps,
            torch_dtype=hidden_states.dtype,
        )

        k_flat, v_flat = self._cache_views(k_cache, v_cache)
        self._padded_buffers(
            padded=padded,
            hidden_size=hidden_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # The kernel addresses X as a flat row-major matrix off its data
        # pointer, so the row pitch must be exactly ``hidden_size``.
        x = self._x_pad
        x[:num_tokens] = hidden_states

        # [T, head_dim] fp32 cos | sin, and the KV cache rows for this batch.
        # Both are the batch's, not the layer's, so a forward builds them once.
        cos_rows, kv_row = self._forward_rows(
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            out_cache_loc=out_cache_loc,
            num_tokens=num_tokens,
            padded=padded,
            device=hidden_states.device,
        )

        with trace_tpu_kernel(
            self._kernel_trace_name,
            tokens=num_tokens,
            padded=padded,
            block_m=block_m,
        ):
            q = kernel(
                x,
                self._projection_weight(),
                cos_rows,
                self._gamma_table(),
                kv_row,
                k_flat,
                v_flat,
            )
        # Hand back the whole padded tile, not ``q[:num_tokens]``.  The paged
        # attention kernel reads each request's query tile at its true row
        # offset with no clamp, so it needs a buffer padded to at least the
        # batch's tile either way: ``_padded_query`` would copy the batch into
        # its own zero-filled buffer, which at prefill is a whole ``[T, 2048]``
        # round trip per layer for nothing.  Given the kernel's own buffer it
        # uses it as it lies.  The padded rows are finite -- ``X``'s pad rows
        # are zero and never rewritten -- and their attention outputs go to
        # output rows the scatter never reads, so the caller only has to trim:
        # ``q[:num_tokens]`` is a view, one op per layer where the slice,
        # reshape and pad-copy this replaces were four.
        return q
