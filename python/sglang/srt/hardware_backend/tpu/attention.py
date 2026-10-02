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
# ``tilelang.tpu.kernels.attention.flash_attention_gqa``.
_FLASH_BLOCK_M = 8
_FLASH_BLOCK_N = 128

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


def _plan_attention_requests(
    *,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    q_lens: list[int],
    encoder_lens: Optional[torch.Tensor],
    is_cross_attn: bool,
) -> list[_AttnReqPlan]:
    """Resolve each request's query rows and KV extent.

    ``q_lens`` is ``extend_seq_lens`` for extend and all-ones for decode. The KV
    window is the request's encoder slice for cross-attention, the sequence
    cached after the encoder prefix otherwise, and the plain sequence with no
    encoder.
    """
    plans = []
    q_start = 0
    for seq_idx in range(len(q_lens)):
        q_len = q_lens[seq_idx]
        seq_len_kv = int(seq_lens[seq_idx])
        if encoder_lens is not None:
            if is_cross_attn:
                kv_start = 0
                kv_len = int(encoder_lens[seq_idx])
            else:
                kv_start = int(encoder_lens[seq_idx])
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
                req_pool_idx=int(req_pool_indices[seq_idx]),
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
    single build serves every request. Only ``d`` (head dim, which sizes the
    on-chip tiles), ``scaling`` (baked as ``sm_scale``) and ``dtype`` (sizes the
    global-buffer element type) are compile-time, so the cache key is
    ``(d, scaling, dtype)`` — ``hq``/``sq``/``skv``/``hkv`` are accepted for
    call-site symmetry but do not trigger a rebuild.

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
    key = (d, float(scaling), _dtype_name(dtype))
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
                d=d, sm_scale=float(scaling), dtype=_dtype_name(dtype)
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


def _resolve_call_device(kernel):
    """Return the kernel's ``call_device`` (device-pointer launch), or None.

    ``_get_flash_kernel`` returns a tilelang ``JITKernel``, which does not expose
    ``call_device`` directly -- it lives on the underlying ``TPUKernel`` and is
    forwarded by the ``PPLKernelAdapter`` at ``kernel.adapter``.  Checking the
    adapter is what keeps the no-copy fast path reachable; without this the
    batched launch silently falls back to the host-tensor path (device pack,
    then ``.cpu()`` round trip), which is the slow path this whole change exists
    to avoid.  Returns None for mock/test kernels that have neither.
    """
    fn = getattr(kernel, "call_device", None)
    if fn is not None:
        return fn
    adapter = getattr(kernel, "adapter", None)
    return getattr(adapter, "call_device", None)


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
    def _build_batched_additive_mask(
        *,
        q_lens: list[int],
        kv_lens: list[int],
        q_pad: int,
        kv_pad: int,
        causal: bool,
        sliding_window_size: Optional[int],
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
        """
        assert len(q_lens) == len(kv_lens)
        masks = torch.full(
            (len(q_lens), q_pad, kv_pad), _FLASH_MASK_NEG, dtype=torch.float32
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
                )
                # Clamp once here rather than on every copy (a fresh tensor per
                # assignment otherwise).
                if per_req is not None:
                    per_req = per_req.clamp_min(_FLASH_MASK_NEG)
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
        act_dtype = _flash_dtype_of(q)

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
            dtype=act_dtype,
        )
        if kernel is None:
            return False
        _record_flash_shape(
            hq=num_q_heads, hkv=num_kv_heads, q_len=q_len, kv_len=kv_len,
            q_pad=q_pad, kv_pad=kv_pad, d=d, scaling=scaling,
        )
        try:
            # [H,S,D] -> [1,H,s_pad,D] host, zero-padded to the kernel tiles and
            # in the kernel's own dtype. The host casts (a plain torch op) fold
            # bf16->fp16 etc. here, where they are free, instead of on-device
            # where the PPL copy dispatcher has no bf16<->fp16 pair.
            def _pad(x, n_heads, s_pad):  # [H,S,D] -> [1,H,s_pad,D]
                padded = torch.zeros(
                    1, n_heads, s_pad, d, dtype=act_dtype,
                )
                x_cpu = x.detach().cpu()
                if x_cpu.dtype != padded.dtype:
                    x_cpu = x_cpu.float().to(padded.dtype)
                padded[0, :, : x.shape[1]] = x_cpu
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
            _record_flash_launch(
                batch=1, q_pad=q_pad, kv_pad=kv_pad, d=d, scaling=scaling
            )
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

        The device-pointer fast path (``kernel.call_device``) is tried first:
        q/k/v are packed on-device via ``_pack_flash_batch_device``, the mask is
        uploaded once, and raw device addresses are passed directly to the PPL
        kernel — no per-request H2D copies.  If the kernel object does not
        expose ``call_device`` (e.g. during autotuning or test mocking), the
        original host-tensor ``kernel(...)`` call is used as a fallback.
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
            mask_cpu = TpuAttnBackend._build_batched_additive_mask(
                q_lens=q_lens,
                kv_lens=kv_lens,
                q_pad=q_pad,
                kv_pad=kv_pad,
                causal=causal,
                sliding_window_size=sliding_window_size,
            ).contiguous()  # [B, q_pad, kv_pad] fp32, on CPU
            # Upload mask once: [B, q_pad, kv_pad] fp32 → TPU device. The mask
            # is fp32 in both kernels (accum_dtype), so no conversion.
            dev = q_b.device
            mask = mask_cpu.to(dev)

            _sync_rt_device()
            dev_call = _resolve_call_device(kernel)
            if dev_call is not None:
                # All tensors already on TPU; pass device pointers directly.
                out = dev_call(q_b, k_b, v_b, mask)
            else:
                # Fallback: host-tensor path (backward compat / test mocking).
                out = kernel(q_b.cpu(), k_b.cpu(), v_b.cpu(), mask_cpu)

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
            # [B, H_q, q_pad, D] result.  ``out`` may be on the TPU device
            # (call_device path, fp16) or on the host (legacy fallback);
            # ``_device_cast`` handles the fp16->bf16 hop the device cannot do
            # directly, and the .to(device) normalises the host case.
            seg = out[b, :, : plan.q_len, :]
            if seg.device.type == "tpu":
                seg = _device_cast(seg, output.dtype)
            else:
                seg = seg.to(output.dtype)
            self._scatter_plan_output(
                output=output,
                plan=plan,
                per_req_out=seg.to(output.device),
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

        plans = _plan_attention_requests(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            q_lens=q_lens,
            encoder_lens=encoder_lens,
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
        for bucket in buckets:
            for start in range(0, len(bucket), _FLASH_MAX_BATCH):
                chunk = bucket[start : start + _FLASH_MAX_BATCH]
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
            q_lens=[int(x) for x in extend_seq_lens],
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
