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

import msgspec
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

# The tilelang kernel's fixed on-chip tiles: block_M query rows, block_N keys.
# It does not clamp partial tiles, so every tile it loads must be fully in range
# -- hence the padding to these multiples. Keep in sync with
# ``tilelang.tpu.kernels.attention.flash_attention_gqa``: these must equal its
# ``block_M`` / ``block_N`` defaults, or a launch reads past the packed buffer.
_FLASH_BLOCK_M = 64
_FLASH_BLOCK_N = 256

# Cap on how many requests one batched launch covers. The kernel itself is happy
# with any B, but the driver materializes [B, H, S_pad, D] fp16 host buffers for
# q/k/v, so a large bucket costs host memory (a 64-wide prefill bucket at
# S_pad=256, H=16, D=128 is ~270 MB). Buckets larger than this are chunked.
_FLASH_MAX_BATCH = 64


def _flash_pad_lens(q_len: int, kv_len: int) -> tuple[int, int]:
    """Pad (q_len, kv_len) up to the kernel's tile multiples.

    ``kv_pad`` is additionally floored at ``q_pad`` because the mask builder
    needs ``query_offset = kv_len - q_len`` to be non-negative.
    """
    q_pad = -(-q_len // _FLASH_BLOCK_M) * _FLASH_BLOCK_M
    kv_pad = -(-kv_len // _FLASH_BLOCK_N) * _FLASH_BLOCK_N
    return q_pad, max(q_pad, kv_pad)


def _flash_dtype_of(x: torch.Tensor) -> torch.dtype:
    """The dtype the flash kernel should be built in, given an activation.

    The kernel is compiled once per dtype, so this must be derived from the
    model's actual activations (Qwen3-0.6B is bf16).  Anything other than fp16
    falls back to fp16, which is the kernel's long-standing default.
    """
    return x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float16


def _to_flash_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast a device tensor to the flash kernel's dtype.

    The device's PPL ``CopyFrom`` dispatcher has no registered bf16<->fp16
    pair (``bf16 -> fp16`` fails with "unsupported dtype pair"), but fp32 sits
    at the hub of the pairs it does support: bf16->fp32, fp32->fp16, and the
    reverse all work.  Routing through fp32 is two supported conversions
    instead of one unsupported one.

    With the kernel built in the model's own dtype (bf16 for Qwen3) this is a
    no-op and no conversion happens at all; it stays as the guard for the fp16
    kernel path.
    """
    if x.dtype == dtype:
        return x
    if {x.dtype, dtype} == {torch.bfloat16, torch.float16}:
        return x.float().to(dtype)
    return x.to(dtype)


def _device_cast(x: torch.Tensor, dtype: torch.dtype) -> torch.dtype:
    """Cast a device tensor to ``dtype``, routing bf16<->fp16 through fp32.

    Same constraint as :func:`_to_flash_dtype`; kept as a distinct name because
    the call sites read differently (kernel output -> model dtype vs. gathered
    q/k/v -> kernel dtype).
    """
    return _to_flash_dtype(x, dtype)


def _pack_flash_batch_device(
    items: list[torch.Tensor], *, n_heads: int, s_pad: int, dtype: torch.dtype
) -> torch.Tensor:
    """Zero-pad a list of per-request ``[H, S_i, D]`` on-device, no H2D/D2H.

    Allocates ``[N, H, s_pad, D]`` in ``dtype`` on the same device as
    ``items[0]``, then copies each request's real rows in place with a
    device-side slice assignment.  All work stays on the TPU; nothing touches
    the host.

    ``dtype`` is the flash kernel's buffer dtype, so when it matches the model
    activations (bf16) every assignment is a plain same-dtype strided copy.
    """
    dev = items[0].device
    d = items[0].shape[-1]
    packed = torch.zeros(
        len(items), n_heads, s_pad, d, dtype=dtype, device=dev
    )
    for i, x in enumerate(items):
        s_real = x.shape[1]
        packed[i, :, :s_real, :] = _to_flash_dtype(x, dtype)
    return packed


# Module-cached flash kernels keyed by (D, scaling). The kernel is shape-generic
# (dynamic B/Hq/Sq/Hkv/Skv), so one build per (head-dim, scale) serves every
# request; building invokes ppl-compile (seconds), so it runs at most once.
_FLASH_KERNELS: dict[tuple, object] = {}

# Same idea for the paged/varlen kernel, which carries its own cache key: it
# reads the KV cache in place, so the kernel's element type must equal the
# cache's and its own arg list differs from the gathered kernel's.
_PAGED_FLASH_KERNELS: dict[tuple, object] = {}

# --- TEMP instrumentation: flash-attention call shapes --------------------
# Set SGLANG_TPU_LOG_FLASH_SHAPES=1 to log every launch (shape + batch size) and,
# at process exit, the deduplicated shape union plus the total launch count. The
# launch count is the metric the batched path is judged on: 32 requests in one
# forward should show up as one launch instead of 32.
_FLASH_SHAPE_SET: set = set()
_FLASH_LAUNCH_COUNT = 0
_FLASH_SHAPE_LOG_ARMED = False


def _flash_log_enabled() -> bool:
    return envs.SGLANG_TPU_LOG_FLASH_SHAPES.get()


def _flash_log_arm_dump() -> None:
    """Register the exit-time dump once, lazily."""
    global _FLASH_SHAPE_LOG_ARMED
    if _FLASH_SHAPE_LOG_ARMED:
        return
    import atexit
    import sys

    def _dump() -> None:
        rows = sorted(_FLASH_SHAPE_SET)
        print(
            f"\n[FLASH-SHAPE] === {len(rows)} distinct shapes, "
            f"{_FLASH_LAUNCH_COUNT} launches ===",
            file=sys.stderr,
            flush=True,
        )
        for r in rows:
            print(
                f"[FLASH-SHAPE] hq={r[0]} hkv={r[1]} q_len={r[2]} kv_len={r[3]} "
                f"q_pad={r[4]} kv_pad={r[5]} d={r[6]} scale={r[7]:.6g}",
                file=sys.stderr,
                flush=True,
            )

    atexit.register(_dump)
    _FLASH_SHAPE_LOG_ARMED = True


def _record_flash_shape(*, hq: int, hkv: int, q_len: int, kv_len: int,
                        q_pad: int, kv_pad: int, d: int, scaling: float) -> None:
    """Record one *request's* shape (deduplicated) for the exit-time table."""
    if not _flash_log_enabled():
        return
    _FLASH_SHAPE_SET.add((hq, hkv, q_len, kv_len, q_pad, kv_pad, d, float(scaling)))
    _flash_log_arm_dump()


def _record_flash_launch(*, batch: int, q_pad: int, kv_pad: int, d: int,
                         scaling: float) -> None:
    """Count and log one *kernel launch*, with the number of requests it covered."""
    global _FLASH_LAUNCH_COUNT
    if not _flash_log_enabled():
        return
    import sys

    _FLASH_LAUNCH_COUNT += 1
    print(
        f"[FLASH-LAUNCH] #{_FLASH_LAUNCH_COUNT} batch={batch} "
        f"q_pad={q_pad} kv_pad={kv_pad} d={d} scale={float(scaling):.6g}",
        file=sys.stderr,
        flush=True,
    )
    _flash_log_arm_dump()
# Set once we discover the flash kernel is unavailable (import/build failure) so
# we don't retry ppl-compile on every request.
_FLASH_DISABLED = False


class _AttnReqPlan(msgspec.Struct, frozen=True):
    """Where one request's query and KV live in the paged buffers.

    Resolved once, read-only, so the batching driver never re-derives offsets
    while it is also scattering outputs.
    """

    q_start: int  # offset of this request's query rows in the packed query
    q_len: int
    kv_start: int  # absolute cache position of its first key
    kv_len: int
    req_pool_idx: int  # row of ``req_to_token`` holding its token indices


def _as_int_list(x) -> Optional[list[int]]:
    """Materialize a device tensor's values on the host in ONE transfer.

    ``int(t[i])`` in a loop is a scalar device->host read *per element*, and
    each one synchronizes the device.  On a 32-request prefill that was 64
    separate round trips (~5.5 ms measured) just to learn lengths the host
    already knew.  ``tolist()`` copies the whole vector once.
    """
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return list(x)


def _plan_attention_requests(
    *,
    req_pool_indices,
    seq_lens,
    q_lens: list[int],
    encoder_lens,
    is_cross_attn: bool,
) -> list[_AttnReqPlan]:
    """Resolve each request's query rows and KV extent.

    ``q_lens`` is ``extend_seq_lens`` for extend and all-ones for decode. The KV
    window is the request's encoder slice for cross-attention, the sequence
    cached after the encoder prefix otherwise, and the plain sequence with no
    encoder.

    ``req_pool_indices`` / ``seq_lens`` / ``encoder_lens`` must already be host
    sequences (see :func:`_as_int_list`) -- reading them element-wise off a
    device tensor would sync once per element.
    """
    plans = []
    q_start = 0
    for seq_idx in range(len(q_lens)):
        q_len = q_lens[seq_idx]
        seq_len_kv = seq_lens[seq_idx]
        if encoder_lens is not None:
            if is_cross_attn:
                kv_start = 0
                kv_len = encoder_lens[seq_idx]
            else:
                kv_start = encoder_lens[seq_idx]
                kv_len = seq_len_kv
        else:
            kv_start = 0
            kv_len = seq_len_kv
        plans.append(
            _AttnReqPlan(
                q_start=q_start,
                q_len=q_len,
                kv_start=kv_start,
                kv_len=kv_len,
                req_pool_idx=req_pool_indices[seq_idx],
            )
        )
        q_start += q_len
    return plans


def _bucket_attention_plans(
    plans: list[_AttnReqPlan], *, n_q_heads: int, n_kv_heads: int
) -> list[list[_AttnReqPlan]]:
    """Group requests that can share one launch, preserving arrival order.

    The bucket key is the *padded* shape the kernel will actually be launched
    with -- padding to the on-chip tile sizes is what makes requests of
    different lengths incompatible, so two requests are only combinable when
    their padded shapes match exactly. Head counts are part of the key because a
    single launch covers one (H_q, H_kv) pair.
    """
    buckets: dict[tuple, list[_AttnReqPlan]] = {}
    for plan in plans:
        q_pad, kv_pad = _flash_pad_lens(plan.q_len, plan.kv_len)
        key = (q_pad, kv_pad, n_q_heads, n_kv_heads)
        buckets.setdefault(key, []).append(plan)
    return list(buckets.values())


def _dtype_name(dtype: torch.dtype) -> str:
    """torch dtype -> the bare name the tilelang factory parses.

    ``str(torch.bfloat16)`` is ``"torch.bfloat16"``, which ``T.dtype`` rejects
    with "unknown dtype"; it wants ``"bfloat16"``.
    """
    return str(dtype).replace("torch.", "")


def _get_flash_kernel(
    *, hq: int, sq: int, skv: int, d: int, hkv: int, scaling: float,
    dtype: torch.dtype = torch.float16,
):
    """Return a cached tilelang flash kernel, or None if tilelang is unavailable.

    The DSL kernel is **shape-generic**: ``B/Hq/Sq/Hkv/Skv`` are dynamic, so a
    single build serves every request. Only ``d`` (head dim), ``scaling`` (baked
    as ``sm_scale``), ``dtype`` (global-buffer element type) and the tile sizes
    are compile-time, so the cache key is ``(d, scaling, dtype)`` — hq/sq/skv/
    hkv are accepted for call-site symmetry but do not trigger a rebuild.

    The tile sizes are passed explicitly (rather than relying on the kernel's
    defaults) so the padding in :func:`_flash_pad_lens` and the kernel's tile
    extents cannot drift apart. They historically did: the kernel moved to
    64/256 while this module still padded to 8/128, which made the kernel read a
    256-key tile out of a 128-key buffer and produce NaN.

    ``dtype`` should be the model's activation dtype (Qwen3-0.6B is bf16).
    Building the kernel in the model's own dtype removes the bf16<->fp16
    conversions at the call boundary entirely, which matters because the device
    has no registered PPL copy for that pair — see :func:`_device_cast`.

    ``sq``/``skv`` must still be padded to the kernel's tile sizes (see
    :meth:`TpuAttnBackend._attend_flash`); the DSL kernel does not clamp partial
    tiles, so every tile it loads must be fully in range.
    """
    global _FLASH_DISABLED
    if _FLASH_DISABLED:
        return None
    block_m, block_n = _FLASH_BLOCK_M, _FLASH_BLOCK_N
    key = (d, float(scaling), _dtype_name(dtype), block_m, block_n)
    kern = _FLASH_KERNELS.get(key)
    if kern is None:
        try:
            import sys

            if "/workspace/tilelang" not in sys.path:
                sys.path.insert(0, "/workspace/tilelang")
            from tilelang.tpu.kernels.attention import flash_attention_gqa

            # Lazy factory call (dynamic-shape idiom): compiles one generic
            # kernel; runtime dims are resolved per call from the input shapes.
            kern = flash_attention_gqa(
                d=d,
                block_M=block_m,
                block_N=block_n,
                sm_scale=float(scaling),
                dtype=_dtype_name(dtype),
            )
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


def _get_paged_flash_kernel(
    *, d: int, scaling: float, dtype: torch.dtype = torch.float16
):
    """Return a cached paged/varlen flash kernel, or None if unavailable.

    The kernel gathers its own K/V rows out of the KV cache and takes the packed
    ``[T, H_q, D]`` queries as they are, so the driver no longer materializes
    per-request ``[H, S_pad, D]`` copies of q/k/v: the only per-request work left
    is building the token index and copying each request's real output rows out
    of the padded per-request slabs.

    ``d``, ``scaling`` and ``dtype`` are the compile-time parameters (``dtype``
    sizes the global buffers, so it must equal the KV cache's element type);
    batch/seq lengths are dynamic. See
    ``tilelang.tpu.kernels.attention_paged.flash_attention_paged``.
    """
    block_m, block_n = _FLASH_BLOCK_M, _FLASH_BLOCK_N
    key = (d, float(scaling), _dtype_name(dtype), block_m, block_n)
    kern = _PAGED_FLASH_KERNELS.get(key)
    if kern is not None:
        return kern
    try:
        import sys

        if "/workspace/tilelang" not in sys.path:
            sys.path.insert(0, "/workspace/tilelang")
        from tilelang.tpu.kernels.attention_paged import flash_attention_paged

        kern = flash_attention_paged(
            d=d,
            block_M=block_m,
            block_N=block_n,
            sm_scale=float(scaling),
            dtype=_dtype_name(dtype),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[TPU] paged flash-attention kernel unavailable (%s); "
            "falling back to the gathered batch path.",
            exc,
        )
        return None
    _PAGED_FLASH_KERNELS[key] = kern
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
        device,
    ) -> Optional[torch.Tensor]:
        """Build an additive [q_len, kv_len] float mask on ``device``, or None.

        Query token ``i`` sits at absolute position ``query_offset + i`` where
        ``query_offset = kv_len - q_len`` (the extend tokens are the tail of the
        sequence; for decode ``q_len == 1``). A key ``j`` is disallowed (masked
        to ``_FLASH_MASK_NEG``) when it is in the future (``j > pos``) or, for
        sliding window, older than the window (``j < pos - window``).

        Built on the target device: the whole construction is elementwise
        indexing/compare/fill, so it never needs host memory.  The fill value is
        already fp16-safe, so callers can pass the result straight to the flash
        kernel.
        """
        has_window = sliding_window_size is not None and sliding_window_size > -1
        if not causal and not has_window:
            return None

        query_offset = kv_len - q_len
        # [q_len, 1] query positions, [1, kv_len] key indices, both expanded to
        # the full [q_len, kv_len] grid *before* comparing.
        #
        # The explicit ``expand`` is load-bearing: this torch_tpu build swaps the
        # operands of a *broadcasting* comparison, so ``j > pos`` with a [q,1]
        # and a [1,kv] operand evaluates as ``pos > j`` and the causal mask comes
        # out transposed (and, when q_len != kv_len, as unrelated garbage).
        # ``+``/``*`` broadcast correctly; only comparisons are affected.
        # Comparing two equal-shaped tensors takes the correct path.
        pos = torch.arange(q_len, device=device).unsqueeze(1) + query_offset
        pos = pos.expand(q_len, kv_len)
        j = torch.arange(kv_len, device=device).unsqueeze(0).expand(q_len, kv_len)

        disallowed = torch.zeros(q_len, kv_len, dtype=torch.bool, device=device)
        if causal:
            disallowed |= j > pos
        if has_window:
            disallowed |= j < pos - sliding_window_size

        mask = torch.zeros(q_len, kv_len, dtype=torch.float32, device=device)
        # ``_FLASH_MASK_NEG``, not ``_MASK_NEG``: -1e9 overflows fp16, and the
        # flash kernel computes in fp16 internally, so a raw -1e9 reaches it as
        # -inf and the softmax produces NaN. Clamping here (rather than at each
        # call site) covers both the aligned and the padded paths.
        mask.masked_fill_(disallowed, _FLASH_MASK_NEG)
        return mask

    @staticmethod
    def _build_batched_additive_mask(
        *,
        q_lens: list[int],
        kv_lens: list[int],
        q_pad: int,
        kv_pad: int,
        causal: bool,
        sliding_window_size: Optional[int],
        device,
    ) -> torch.Tensor:
        """Assemble the block-diagonal ``[B, q_pad, kv_pad]`` additive mask.

        Slice ``b`` is request ``b``'s own padded mask, so a query row can only
        put softmax mass on its own request's keys. The per-request causal/window
        arithmetic is delegated to :meth:`_build_additive_mask` -- this only pads
        and stacks. Padded *key columns* stay at ``_FLASH_MASK_NEG``: their V rows
        are meaningless zeros, so a real query must never attend them. Padded
        *query rows* are filled with 0 instead, which is neutral either way (the
        kernel's softmax max is reduced per query row -- ``reduce_max(acc_s,
        m_cur, dim=1)`` -- so a padded row cannot perturb a real one) and keeps
        the padded rows' arithmetic away from the masked-exponent path. Those
        rows' outputs are sliced off regardless.

        Built entirely on ``device``: there is no host-side assembly and no
        upload, so the mask never crosses the PCIe boundary.
        """
        assert len(q_lens) == len(kv_lens)
        masks = torch.full(
            (len(q_lens), q_pad, kv_pad),
            _FLASH_MASK_NEG,
            dtype=torch.float32,
            device=device,
        )
        # Requests in one bucket usually share (q_len, kv_len) -- in homogeneous
        # prefill all of them do -- so the per-request block is often identical.
        # Build each distinct block once and reuse it; rebuilding it per request
        # was 31/32 redundant work on the benchmarked prefill (16.8 ms measured
        # for a mask that only has one distinct value). ``causal`` and the window
        # are constant across the call, so the key is just (q_len, kv_len).
        cache: dict[tuple[int, int], Optional[torch.Tensor]] = {}
        for b, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
            key = (q_len, kv_len)
            if key not in cache:
                per_req = TpuAttnBackend._build_additive_mask(
                    q_len=q_len,
                    kv_len=kv_len,
                    causal=causal,
                    sliding_window_size=sliding_window_size,
                    device=device,
                )
                cache[key] = per_req
            per_req = cache[key]
            # ``None`` means "no constraint at all" (decode attending every
            # cached key), i.e. an all-zero real block.
            if per_req is not None:
                masks[b, :q_len, :kv_len] = per_req
            else:
                masks[b, :q_len, :kv_len] = 0.0
            masks[b, q_len:, :] = 0.0
        return masks

    @staticmethod
    def _attend_flash(
        *,
        q: torch.Tensor,  # [H_q, S_q, D]
        k: torch.Tensor,  # [H_kv, S_kv, D]
        v: torch.Tensor,  # [H_kv, S_kv, D]
        out: torch.Tensor,  # [H_q, S_q, D] (written in place)
        scaling: float,
        cpu_mask: Optional[torch.Tensor],  # [S_q, S_kv] additive on-device, or None
    ) -> bool:
        """Try the tilelang PPL flash-attention kernel. Returns True on success
        (``out`` written), False to fall back to the per-head matmul loop.

        The DSL kernel has fixed on-chip tiles (``_FLASH_BLOCK_M`` rows of
        queries, ``_FLASH_BLOCK_N`` keys) and no partial-tile clamping, so
        ``S_q``/``S_kv`` are padded up to those multiples here and sliced back
        afterwards. Padded query rows and padded keys are written as zeros in
        q/k/v; the mask's padded *key columns* are set to ``-30000`` so a real
        query never puts softmax mass on a padded key (whose V is meaningless).
        Padded *query rows* need no masking -- their outputs are sliced off.

        Everything stays on the device: q/k/v are packed with
        :func:`_pack_flash_batch_device` and the kernel launches with device
        pointers, so no tensor crosses the PCIe boundary.
        """
        num_q_heads, q_len, d = q.shape
        num_kv_heads, kv_len, _ = k.shape
        act_dtype = _flash_dtype_of(q)

        block_m, block_n = _FLASH_BLOCK_M, _FLASH_BLOCK_N
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
            dtype=act_dtype,
        )
        if kernel is None:
            return False
        _record_flash_shape(
            hq=num_q_heads, hkv=num_kv_heads, q_len=q_len, kv_len=kv_len,
            q_pad=q_pad, kv_pad=kv_pad, d=d, scaling=scaling,
        )
        try:
            # [H,S,D] -> [1,H,s_pad,D], zero-padded to the kernel tiles and in
            # the kernel's own dtype, entirely on-device.
            q_b = _pack_flash_batch_device(
                [q], n_heads=num_q_heads, s_pad=q_pad, dtype=act_dtype
            )
            k_b = _pack_flash_batch_device(
                [k], n_heads=num_kv_heads, s_pad=kv_pad, dtype=act_dtype
            )
            v_b = _pack_flash_batch_device(
                [v], n_heads=num_kv_heads, s_pad=kv_pad, dtype=act_dtype
            )

            device = out.device
            if cpu_mask is not None:
                m = _device_cast(cpu_mask.float(), torch.float32).clamp_min(
                    _FLASH_MASK_NEG
                )
            else:
                # No constraint at all (decode attends every cached key): the
                # real block is all-zeros, and the padded key columns below are
                # masked off.
                m = torch.zeros(q_len, kv_len, dtype=torch.float32, device=device)

            if q_pad != q_len or kv_pad != kv_len:
                # Mask row r <-> query row r, column j <-> key j (the kernel
                # bakes in no causal constraint). Padding starts at -30000 so
                # the added key columns are masked out; the padded query rows
                # are overwritten with 0 because the kernel's softmax is taken
                # over a whole query tile -- an all-(-30000) row would make the
                # tile-wide max -30000 and blank the real rows beside it.
                padded = torch.full(
                    (q_pad, kv_pad),
                    _FLASH_MASK_NEG,
                    dtype=torch.float32,
                    device=device,
                )
                padded[:q_len, :kv_len] = m
                padded[q_len:, :] = 0.0
                m = padded
            mask = m.unsqueeze(0).contiguous()  # [1, S_q, S_kv]

            # The kernel launches with device addresses; run_device_ptrs
            # re-asserts the PPL runtime's device immediately before the launch,
            # so no caller-side _sync_rt_device is needed.
            res = kernel(q_b, k_b, v_b, mask)  # [1,Hq,q_pad,D] on device
            _record_flash_launch(
                batch=1, q_pad=q_pad, kv_pad=kv_pad, d=d, scaling=scaling
            )
            # [1,Hq,q_pad,D] -> [Hq,q_len,D], still on device.
            resc = res[0, :, :q_len].contiguous()
            out.copy_(_device_cast(resc, out.dtype))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[TPU] flash-attention launch failed (%s); "
                "falling back to per-head matmul loop.",
                exc,
            )
            return False

    @staticmethod
    def _attend_flash_batch(
        *,
        qs: list[torch.Tensor],  # per request: [H_q, S_q, D] on-device
        ks: list[torch.Tensor],  # per request: [H_kv, S_kv, D] on-device
        vs: list[torch.Tensor],  # per request: [H_kv, S_kv, D] on-device
        q_lens: list[int],
        kv_lens: list[int],
        q_pad: int,
        kv_pad: int,
        scaling: float,
        causal: bool,
        sliding_window_size: Optional[int],
    ) -> Optional[torch.Tensor]:
        """One flash-attention launch for a whole bucket of requests.

        Returns the padded ``[B, H_q, q_pad, D]`` fp16 result on the TPU device,
        or None if the launch failed.  All-or-nothing: the caller falls back to
        the per-request loop for the *whole* bucket so no partially-batched
        result is ever mixed into the output.

        q/k/v are packed on-device via ``_pack_flash_batch_device``, the mask is
        built on-device, and raw device addresses are passed straight to the PPL
        kernel — no per-request H2D copies and no host marshalling anywhere.
        """
        num_q_heads, _, d = qs[0].shape
        num_kv_heads = ks[0].shape[0]
        act_dtype = _flash_dtype_of(qs[0])
        kernel = _get_flash_kernel(
            hq=num_q_heads,
            sq=q_pad,
            skv=kv_pad,
            d=d,
            hkv=num_kv_heads,
            scaling=scaling,
            dtype=act_dtype,
        )
        if kernel is None:
            return None
        try:
            # --- device-pointer fast path -----------------------------------
            # Pack q/k/v entirely on-device (zero-pad to padded shape, no H2D).
            # Built in the kernel's own dtype, so when it matches the model
            # activations every assignment is a same-dtype strided copy and no
            # conversion happens at all.
            q_b = _pack_flash_batch_device(
                qs, n_heads=num_q_heads, s_pad=q_pad, dtype=act_dtype
            )
            k_b = _pack_flash_batch_device(
                ks, n_heads=num_kv_heads, s_pad=kv_pad, dtype=act_dtype
            )
            v_b = _pack_flash_batch_device(
                vs, n_heads=num_kv_heads, s_pad=kv_pad, dtype=act_dtype
            )
            # Mask is built on the device and passed by pointer — no host
            # assembly, no upload.  It is fp32 in both kernels (accum_dtype).
            mask = TpuAttnBackend._build_batched_additive_mask(
                q_lens=q_lens,
                kv_lens=kv_lens,
                q_pad=q_pad,
                kv_pad=kv_pad,
                causal=causal,
                sliding_window_size=sliding_window_size,
                device=q_b.device,
            ).contiguous()  # [B, q_pad, kv_pad] fp32, on device

            # Device pointers only: run_device_ptrs re-asserts the runtime's
            # device right before the launch.
            out = kernel(q_b, k_b, v_b, mask)  # [B, Hq, q_pad, D] on device

            _record_flash_launch(
                batch=len(qs), q_pad=q_pad, kv_pad=kv_pad, d=d, scaling=scaling
            )
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[TPU] batched flash-attention launch failed (%s); "
                "falling back to the per-request loop for all %d requests.",
                exc,
                len(qs),
            )
            return None

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

        # Device-resident mask: the flash path consumes it directly, and the
        # matmul fallback below adds it on-device too.  No host round trip.
        mask = self._build_additive_mask(
            q_len=q_len,
            kv_len=kv_len,
            causal=causal,
            sliding_window_size=sliding_window_size,
            device=q.device,
        )

        if envs.SGLANG_TPU_USE_FLASH_ATTN.get() and self._attend_flash(
            q=q, k=k, v=v, out=out, scaling=scaling, cpu_mask=mask
        ):
            return

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
    #
    # The base class loops one request at a time and launches attention once
    # per iteration. Here the loop is *planned* first -- every request's query
    # rows and KV extent resolved read-only into _AttnReqPlan -- then requests
    # sharing a padded shape are handed to the flash kernel in one launch. Any
    # request that cannot be grouped, or whose bucket fails to launch, runs the
    # original per-request path unchanged, so batching only ever removes
    # launches; it never changes which math a request gets.

    @staticmethod
    def _use_batched_flash(*, is_cross_attn: bool) -> bool:
        """Batched launches need an additive mask, not an implicit causal one.

        Cross-attention is left on the per-request path: it is rare here and its
        KV extent comes from the encoder, so batching it buys little.
        """
        if is_cross_attn:
            return False
        return (
            envs.SGLANG_TPU_USE_FLASH_ATTN.get()
            and envs.SGLANG_TPU_USE_BATCHED_FLASH_ATTN.get()
        )

    def _gather_plan_qkv(
        self,
        *,
        plan: _AttnReqPlan,
        query: torch.Tensor,  # [H_q, num_tokens, D]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slice one request's q/k/v out of the packed query and paged KV cache."""
        q_i = query[:, plan.q_start : plan.q_start + plan.q_len, :]
        tokens = req_to_token[
            plan.req_pool_idx, plan.kv_start : plan.kv_start + plan.kv_len
        ]
        k_i = k_cache[tokens].movedim(0, 1)  # [H_kv, S_kv, D]
        v_i = v_cache[tokens].movedim(0, 1)
        if not (q_i.dtype == k_i.dtype == v_i.dtype):
            k_i = k_i.to(q_i.dtype)
            v_i = v_i.to(q_i.dtype)
        return q_i, k_i, v_i

    @staticmethod
    def _scatter_plan_output(
        *,
        output: torch.Tensor,  # [num_tokens, H_q, D]
        plan: _AttnReqPlan,
        per_req_out: torch.Tensor,  # [H_q, S_q, D]
    ) -> None:
        """Write one request's [H_q, S_q, D] result back into [T, H_q, D] layout."""
        output[plan.q_start : plan.q_start + plan.q_len, :, :] = per_req_out.movedim(
            1, 0
        )

    def _attend_plan_serial(
        self,
        *,
        plan: _AttnReqPlan,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        scaling: float,
        causal: bool,
        sliding_window_size: Optional[int],
    ) -> None:
        """The fallback: one request, one launch, exactly the pre-batching path."""
        q_i, k_i, v_i = self._gather_plan_qkv(
            plan=plan,
            query=query,
            k_cache=k_cache,
            v_cache=v_cache,
            req_to_token=req_to_token,
        )
        per_req_out = torch.empty_like(q_i)
        self._attend_per_req(
            q=q_i,
            k=k_i,
            v=v_i,
            out=per_req_out,
            scaling=scaling,
            causal=causal,
            sliding_window_size=sliding_window_size,
        )
        self._scatter_plan_output(output=output, plan=plan, per_req_out=per_req_out)

    def _attend_plan_bucket(
        self,
        *,
        plans: list[_AttnReqPlan],
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        scaling: float,
        causal: bool,
        sliding_window_size: Optional[int],
    ) -> bool:
        """One launch for every request in ``plans`` (same padded shape).

        Returns False if the launch failed, leaving ``output`` untouched so the
        caller can redo the whole bucket on the per-request path.
        """
        q_lens = [p.q_len for p in plans]
        kv_lens = [p.kv_len for p in plans]
        # Every plan here shares a padded shape by construction (the bucket key),
        # so one request's padding describes the whole launch.
        q_pad, kv_pad = _flash_pad_lens(q_lens[0], kv_lens[0])
        d = query.shape[-1]
        for plan in plans:
            _record_flash_shape(
                hq=query.shape[0], hkv=k_cache.shape[1],
                q_len=plan.q_len, kv_len=plan.kv_len,
                q_pad=q_pad, kv_pad=kv_pad, d=d, scaling=scaling,
            )

        qs, ks, vs = [], [], []
        for plan in plans:
            q_i, k_i, v_i = self._gather_plan_qkv(
                plan=plan,
                query=query,
                k_cache=k_cache,
                v_cache=v_cache,
                req_to_token=req_to_token,
            )
            qs.append(q_i)
            ks.append(k_i)
            vs.append(v_i)

        out = self._attend_flash_batch(
            qs=qs,
            ks=ks,
            vs=vs,
            q_lens=q_lens,
            kv_lens=kv_lens,
            q_pad=q_pad,
            kv_pad=kv_pad,
            scaling=scaling,
            causal=causal,
            sliding_window_size=sliding_window_size,
        )
        if out is None:
            return False

        for b, plan in enumerate(plans):
            # Slice this request's real rows: [H_q, q_len, D] from the padded
            # [B, H_q, q_pad, D] result. Both are device tensors;
            # ``_device_cast`` handles the fp16->bf16 hop the device cannot do
            # directly.
            seg = _device_cast(out[b, :, : plan.q_len, :], output.dtype)
            self._scatter_plan_output(
                output=output,
                plan=plan,
                per_req_out=seg,
            )
        return True

    @staticmethod
    def _build_paged_token_index(
        *,
        plans: list[_AttnReqPlan],
        req_to_token: torch.Tensor,
        n_kv_heads: int,
        kv_pad: int,
    ) -> torch.Tensor:
        """Build the paged kernel's ``Idx[B, H_kv, kv_pad]`` row index.

        Entry ``(b, h, j)`` is the flat row of the ``[rows * H_kv, D]`` view of
        the KV cache to gather for request ``b``'s ``j``-th key under kv head
        ``h``, i.e. ``slot * H_kv + h`` -- the cache itself is NHD, so its rows
        for one key are adjacent and one gather fetches all heads at once.

        Padded columns (``j >= kv_len``) stay at slot 0, which is a *valid* row:
        the additive mask discards them, and an out-of-range index would fault
        the SDMA gather instead of contributing nothing.
        """
        device = req_to_token.device
        slots = torch.zeros((len(plans), kv_pad), dtype=torch.int32, device=device)
        for b, plan in enumerate(plans):
            slots[b, : plan.kv_len] = req_to_token[
                plan.req_pool_idx, plan.kv_start : plan.kv_start + plan.kv_len
            ]
        heads = torch.arange(n_kv_heads, dtype=torch.int32, device=device)
        # [B, 1, kv_pad] * H_kv + [1, H_kv, 1] -> [B, H_kv, kv_pad]
        idx = slots.unsqueeze(1) * n_kv_heads + heads.view(1, -1, 1)
        return idx.contiguous()

    def _attend_plan_bucket_paged(
        self,
        *,
        plans: list[_AttnReqPlan],
        query: torch.Tensor,  # [H_q, num_tokens, D]
        output: torch.Tensor,  # [num_tokens, H_q, D]
        k_cache: torch.Tensor,  # [rows, H_kv, D] (NHD, unchanged)
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        scaling: float,
        causal: bool,
        sliding_window_size: Optional[int],
    ) -> bool:
        """One paged/varlen launch for ``plans``; False if the path cannot be used.

        The kernel reads K/V straight out of the cache at ``Idx``'s rows and
        takes the packed queries as they lie, so the driver's per-request gather,
        pack and scatter collapse into a token index plus one copy of each
        request's real output rows. Returns False -- leaving ``output``
        untouched -- whenever the layout is not one the kernel can consume, so
        the caller can use the gathered batch path instead.
        """
        # The kernel addresses [T, H_q, D] by a per-request row offset, which is
        # exactly the model's packed query layout: recover it as a view of the
        # [H_q, T, D] tensor this driver carries. A copy here would cost as much
        # as the packing this path is meant to remove, so bail out instead.
        q_packed = query.movedim(0, 1)  # [T, H_q, D]
        if not (q_packed.is_contiguous() and output.is_contiguous()):
            return False
        if not (k_cache.is_contiguous() and v_cache.is_contiguous()):
            return False
        act_dtype = _flash_dtype_of(query)
        if k_cache.dtype != act_dtype or v_cache.dtype != act_dtype:
            return False

        q_lens = [p.q_len for p in plans]
        kv_lens = [p.kv_len for p in plans]
        # Every plan in the bucket shares a padded shape by construction.
        q_pad, kv_pad = _flash_pad_lens(q_lens[0], kv_lens[0])
        d = query.shape[-1]
        kernel = _get_paged_flash_kernel(d=d, scaling=scaling, dtype=act_dtype)
        if kernel is None:
            return False
        for plan in plans:
            _record_flash_shape(
                hq=query.shape[0], hkv=k_cache.shape[1],
                q_len=plan.q_len, kv_len=plan.kv_len,
                q_pad=q_pad, kv_pad=kv_pad, d=d, scaling=scaling,
            )
        # The kernel reads each request's query tile at its true row offset
        # ``q_start + tq*block_M`` with no clamp, so the trailing tile of the
        # last request over-reads by up to ``block_M - 1`` rows past the packed
        # buffer. Pad the buffer so that read stays in bounds (the extra rows
        # feed only discarded output). ``max(q_start) + q_pad`` is the exact
        # upper bound of any read; a short decode batch (fewer rows than one
        # tile) is the common case that needs it.
        max_read = max(plan.q_start for plan in plans) + q_pad
        if q_packed.shape[0] < max_read:
            padded = torch.zeros(
                (max_read, *q_packed.shape[1:]),
                dtype=q_packed.dtype,
                device=q_packed.device,
            )
            padded[: q_packed.shape[0]] = q_packed
            q_packed = padded

        try:
            idx = self._build_paged_token_index(
                plans=plans,
                req_to_token=req_to_token,
                n_kv_heads=k_cache.shape[1],
                kv_pad=kv_pad,
            )
            q_off = torch.tensor(
                [plan.q_start for plan in plans],
                dtype=torch.int32,
                device=q_packed.device,
            )
            mask = self._build_batched_additive_mask(
                q_lens=q_lens,
                kv_lens=kv_lens,
                q_pad=q_pad,
                kv_pad=kv_pad,
                causal=causal,
                sliding_window_size=sliding_window_size,
                device=q_packed.device,
            ).contiguous()  # [B, q_pad, kv_pad] fp32, on device

            out = kernel(q_packed, k_cache, v_cache, idx, mask, q_off)
            _record_flash_launch(
                batch=len(plans), q_pad=q_pad, kv_pad=kv_pad, d=d, scaling=scaling
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[TPU] paged flash-attention launch failed (%s); "
                "falling back to the gathered batch path for all %d requests.",
                exc,
                len(plans),
            )
            return False

        # [B, q_pad, H_q, D] -> the packed [T, H_q, D] output, real rows only.
        for b, plan in enumerate(plans):
            output[plan.q_start : plan.q_start + plan.q_len] = _device_cast(
                out[b, : plan.q_len], output.dtype
            )
        return True

    def _run_sdpa_batched_attention(
        self,
        *,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        q_lens: list[int],
        encoder_lens: Optional[torch.Tensor],
        scaling: float,
        causal: bool,
        is_cross_attn: bool,
        sliding_window_size: Optional[int],
    ) -> torch.Tensor:
        """Shared extend/decode core: plan, bucket, launch, scatter, fall back."""
        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        # One host transfer each, not one per element: the planner only needs
        # these as plain ints, and indexing a device tensor element-wise costs
        # a device sync per request.
        plans = _plan_attention_requests(
            req_pool_indices=_as_int_list(req_pool_indices),
            seq_lens=_as_int_list(seq_lens),
            q_lens=q_lens,
            encoder_lens=_as_int_list(encoder_lens),
            is_cross_attn=is_cross_attn,
        )

        def run_serial(plan_list):
            for plan in plan_list:
                self._attend_plan_serial(
                    plan=plan,
                    query=query,
                    output=output,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    req_to_token=req_to_token,
                    scaling=scaling,
                    causal=causal,
                    sliding_window_size=sliding_window_size,
                )

        if not self._use_batched_flash(is_cross_attn=is_cross_attn):
            run_serial(plans)
            return output

        buckets = _bucket_attention_plans(
            plans, n_q_heads=query.shape[0], n_kv_heads=k_cache.shape[1]
        )
        use_paged = envs.SGLANG_TPU_USE_PAGED_FLASH_ATTN.get()
        for bucket in buckets:
            for start in range(0, len(bucket), _FLASH_MAX_BATCH):
                chunk = bucket[start : start + _FLASH_MAX_BATCH]
                # The paged kernel needs no per-request copies, so it is worth a
                # launch even for a lone request; the gathered path is not.
                if use_paged and self._attend_plan_bucket_paged(
                    plans=chunk,
                    query=query,
                    output=output,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    req_to_token=req_to_token,
                    scaling=scaling,
                    causal=causal,
                    sliding_window_size=sliding_window_size,
                ):
                    continue
                # A lone request has nothing to batch -- skip straight to the
                # path that already handles every shape.
                if len(chunk) == 1:
                    run_serial(chunk)
                    continue
                if not self._attend_plan_bucket(
                    plans=chunk,
                    query=query,
                    output=output,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    req_to_token=req_to_token,
                    scaling=scaling,
                    causal=causal,
                    sliding_window_size=sliding_window_size,
                ):
                    run_serial(chunk)
        return output

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

        return self._run_sdpa_batched_attention(
            query=query,
            output=output,
            k_cache=k_cache,
            v_cache=v_cache,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            q_lens=_as_int_list(extend_seq_lens),
            encoder_lens=encoder_lens,
            scaling=scaling,
            causal=causal,
            is_cross_attn=is_cross_attn,
            sliding_window_size=sliding_window_size,
        )

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
        return self._run_sdpa_batched_attention(
            query=query,
            output=output,
            k_cache=k_cache,
            v_cache=v_cache,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            # One query token per request -- the uniform shape that makes decode
            # bucket into a single launch.
            q_lens=[1] * int(seq_lens.shape[0]),
            encoder_lens=encoder_lens,
            scaling=scaling,
            # A decode step attends all cached keys; the driver passes
            # causal=False (every key is in the past).
            causal=causal,
            is_cross_attn=is_cross_attn,
            sliding_window_size=sliding_window_size,
        )

    def support_triton(self):
        return False
