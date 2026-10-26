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

import torch

#: Compiled kernels, keyed by the full geometry + tile + dtype.  The build is
#: ~1 minute, so the cache is process-wide and the wrap's ``__call__`` is the
#: only place that pays for it.
_KERNELS: dict[tuple, object] = {}

#: Query-row tile in tokens.  ``block_M`` is baked, so a short decode batch gets
#: its own small-tile build instead of running 63/64 of a tile empty.
_BLOCK_M_BIG = 64
_BLOCK_M_SMALL = 16
_BLOCK_K = 256

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
        self._cos_pad: torch.Tensor | None = None
        self._kv_row: torch.Tensor | None = None

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
        """Cached ``X``, ``CosRows`` and ``KVRow`` tiles of ``padded`` rows.

        Rows past the real token count are never rewritten once the buffers
        exist: they hold whatever the last call with a larger token count left
        there, which is safe because a padded row's ``Q`` is sliced off and its
        ``KV`` goes to the pool's reserved spare slot.  The ``KVRow`` tail is the
        one exception -- a stale value there would address a *live* slot, so the
        spare-slot columns are (re)written whenever the buffer is built.
        """
        key = (padded, hidden_size, dtype, device)
        if self._buf_key == key:
            return
        n_kv = self._num_kv_heads
        self._x_pad = torch.zeros((padded, hidden_size), dtype=dtype, device=device)
        self._cos_pad = torch.zeros(
            (padded, self._head_dim), dtype=torch.float32, device=device
        )
        self._kv_row = torch.zeros((2 * n_kv, padded), dtype=torch.int32, device=device)
        rows = torch.arange(2 * n_kv, dtype=torch.int32, device=device)
        # ``_cache_views`` runs first, so the pool's row count is known.
        pad_slot = self._k_flat.shape[0] // (2 * n_kv) - 1
        self._kv_row[:, :] = rows.view(-1, 1) + pad_slot * (2 * n_kv)
        self._buf_key = key

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
        """Return ``q`` as the packed ``[T, num_heads * head_dim]`` bf16 tensor.

        ``K``/``V`` are already in ``k_cache``/``v_cache`` when this returns;
        rows past ``T`` (the tile padding) write to the pool's spare slot only.
        """
        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        block_m = _BLOCK_M_BIG if num_tokens >= _BLOCK_M_BIG else _BLOCK_M_SMALL
        padded = -(-num_tokens // block_m) * block_m

        kernel = _get_kernel(
            hidden_size=hidden_size,
            num_heads=self._num_heads,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
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

        # [T, head_dim] fp32 cos | sin.  The cache itself is bf16 once the
        # assembly path has matched the activation dtype, so the copy widens it
        # -- a bf16->fp32 widen has no on-chip instruction, but this is a plain
        # elementwise op on the driver side, not in the kernel.
        cos_rows = self._cos_pad
        cos_rows[:num_tokens] = cos_sin_cache.index_select(0, positions.flatten())

        kv_row = self._kv_row
        rows = torch.arange(
            2 * self._num_kv_heads, dtype=torch.int32, device=out_cache_loc.device
        )
        # ``out_cache_loc`` is int64; the kernel reads this table as 32-bit
        # integers, so narrow it here -- an int64 table would be read at half its
        # real stride and yield interleaved garbage rows.
        slots = out_cache_loc.view(1, -1).to(torch.int32)
        kv_row[:, :num_tokens] = slots * (2 * self._num_kv_heads) + rows.view(-1, 1)

        q = kernel(x, self._qkv_proj.weight, cos_rows, self._gamma_table(),
                   kv_row, k_flat, v_flat)
        # Same shape contract as ``forward_prepare_native``: the attention
        # backend and ``torch.empty_like(q)`` below it both assume the packed
        # ``[T, num_heads * head_dim]`` layout.
        return q[:num_tokens].reshape(num_tokens, -1)
