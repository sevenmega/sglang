"""Decompose the TPU attention op: where do the 227ms/launch actually go?

Times each phase of the batched flash-attention driver separately:
  gather  -- slicing q/k/v out of the packed query + paged KV cache
  pack    -- zero-padding per request into [B,H,s_pad,D], on-device
  mask    -- building the [B,q_pad,kv_pad] additive mask, on-device
  launch  -- the PPL kernel call (device pointers). Synchronous: the launch
             call itself blocks on py_sync_device, so this phase includes the
             whole kernel execution, not just dispatch.
  scatter -- slicing the output back into [T,H,D]

With ``SGLANG_TPU_USE_PAGED_FLASH_ATTN=1`` (the default) the driver instead
takes one ``paged`` launch that reads the KV cache in place, so
``gather``/``pack`` and ``scatter`` drop out and their place is taken by:
  paged   -- the whole paged launch (queries as-is, cache in place)
  idx     -- the [B,H_kv,kv_pad] token index it gathers by (nested)

Set ``SGLANG_TPU_USE_PAGED_FLASH_ATTN=0`` to profile the gathered path instead.

Each phase is bracketed by a device sync inside ``timed``, so the numbers are
serialized costs and their sum overstates a pipelined run.

The device comes from ``--device-id`` (default: SGLANG_TPU_DEVICE_ID, else 0).
torch_tpu ignores TPU_VISIBLE_DEVICES, so that env var alone will not move it.

Run:
    SGLANG_TPU_DEVICE_ID=6 python3 profile_attn_phases.py --batch-size 32 --input-len 256
"""

from __future__ import annotations

import os
import argparse
import statistics
import sys
import time

import torch

WARMUP = 2
ROUNDS = 5


def ceildiv(a, b):
    return -(-a // b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/Qwen3-0.6B")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--input-len", type=int, default=256)
    ap.add_argument(
        "--device-id",
        type=int,
        default=int(os.environ.get("SGLANG_TPU_DEVICE_ID", "0")),
    )
    ap.add_argument("--profile", action="store_true", help="Enable hardware profiling")
    ap.add_argument("--verbose", action="store_true", help="Verbose profiling output")
    ap.add_argument(
        "--raw",
        action="store_true",
        help="Skip the per-phase wrappers and report only the end-to-end op time. "
             "The wrappers sync twice per call, so an instrumented total cannot "
             "be compared with an uninstrumented one; use this for before/after.",
    )
    args = ap.parse_args()
    DEV = str(args.device_id)

    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.utils import suppress_other_loggers
    import argparse as _ap
    from types import SimpleNamespace

    from sglang.benchmark.one_batch import (
        _maybe_prepare_mlp_sync_batch,
        load_model,
        prepare_synthetic_inputs_for_latency_test,
    )
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    suppress_other_loggers()
    parser = _ap.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    server_args = ServerArgs.from_cli_args(
        parser.parse_args(["--model-path", args.model_path, "--base-gpu-id", DEV])
    )
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)

    bench_runner, _ = load_model(server_args, port_args, args.device_id, 0)
    mr = bench_runner.torch_runner
    dev = mr.device
    SYNC = lambda: torch.get_device_module(dev).synchronize()
    print(f"[ph] device={dev}", flush=True)

    reqs = prepare_synthetic_inputs_for_latency_test(args.batch_size, args.input_len)
    bench_runner.clear()

    class _TreeCache(SimpleNamespace):
        def supports_swa(self): return False
        def supports_mamba(self): return False
        def is_chunk_cache(self): return False
        def is_tree_cache(self): return not self.is_chunk_cache()
        def evict(self, params): pass

    tree = _TreeCache(
        page_size=server_args.page_size, device=dev,
        token_to_kv_pool_allocator=mr.token_to_kv_pool_allocator,
    )
    batch = ScheduleBatch.init_new(
        reqs=reqs, req_to_token_pool=mr.req_to_token_pool,
        token_to_kv_pool_allocator=mr.token_to_kv_pool_allocator,
        tree_cache=tree, model_config=mr.model_config,
        enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, mr)
    if batch.input_ids is None and getattr(batch, "prefill_input_ids_cpu", None) is not None:
        batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
        batch.prefill_input_ids_cpu = None
    fwd = ForwardBatch.init_new(batch, mr, return_hidden_states_before_norm=False)
    ctx = ForwardContext(attn_backend=mr.attn_backend)

    model = mr.model
    layers = model.model.layers if hasattr(model, "model") else model.layers
    layer0 = layers[0]
    attn = layer0.self_attn

    with torch.no_grad(), forward_context(ctx):
        mr.forward(fwd)
    SYNC()

    # Recreate the layer-0 attention inputs.
    hs = model.model.embed_tokens(fwd.input_ids) if hasattr(model, "model") else model.embed_tokens(fwd.input_ids)
    positions = fwd.positions if fwd.positions is not None else torch.arange(hs.shape[0], device=dev, dtype=torch.int64)
    with torch.no_grad(), forward_context(ctx):
        normed = layer0.input_layernorm(hs)
        qkv, _ = attn.qkv_proj(normed)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        # The model runs q/k through qk-norm + RoPE, which re-materialize them as
        # fresh contiguous tensors. A raw ``split`` view is strided (its row
        # pitch is the whole fused qkv row), and the paged path correctly
        # refuses a non-contiguous query rather than copying it. Reproduce the
        # model's layout here so this harness measures what the model actually
        # launches -- once, outside the timed region.
        q = q.contiguous()

    T = q.shape[0]
    Hq, D = attn.num_heads, attn.head_dim
    Hkv = attn.num_kv_heads
    print(f"[ph] T={T} Hq={Hq} Hkv={Hkv} D={D} batch={fwd.batch_size}", flush=True)

    # --- instrument the driver's phases ------------------------------------
    from sglang.srt.hardware_backend.tpu import attention as attn_mod

    backend = mr.attn_backend
    print(f"[ph] backend={type(backend).__name__}", flush=True)

    # Wrap the internal helpers to time them individually.
    #
    # ``mask`` and ``pack`` are called from *inside* ``_attend_flash_batch``, so
    # a naive "time every wrapper and sum" double counts them: they would be
    # reported both on their own and again in ``launch``. Each phase therefore
    # records its EXCLUSIVE time (elapsed minus whatever nested phases charged
    # to it) and charges its own elapsed to the enclosing phase on the way out.
    # ``launch`` then means "the batched driver excluding mask and pack", and
    # the kernel alone is ``launch - mask - pack``.
    phases: dict[str, list[float]] = {}
    calls: dict[str, int] = {}
    _stack: list[list] = []  # [name, t0, child_time]

    def _enter(name: str) -> None:
        # Sync at every boundary: each phase is measured as a SERIALIZED cost
        # (enqueue + execution), not just Python enqueue time. This is what
        # makes the per-phase numbers meaningful -- and it is also why the
        # instrumented e2e is much larger than the uninstrumented one: the
        # ~69 wrapped calls pay ~138 extra device round-trips between them.
        SYNC()
        _stack.append([name, time.perf_counter(), 0.0])

    def _exit() -> None:
        name, t0, child = _stack.pop()
        SYNC()
        elapsed = time.perf_counter() - t0
        phases.setdefault(name, []).append(elapsed - child)
        calls[name] = calls.get(name, 0) + 1
        if _stack:
            _stack[-1][2] += elapsed  # charge this phase to its parent

    def timed(name, fn):
        # ``--raw`` measures the op as it actually runs: no wrappers, no extra
        # device round-trips. The callers below still rebind the (now identical)
        # functions, which is harmless.
        if args.raw:
            return fn

        def wrapper(*a, **kw):
            _enter(name)
            try:
                return fn(*a, **kw)
            finally:
                _exit()

        return wrapper

    cls = type(backend)
    orig_gather = backend._gather_plan_qkv
    orig_mask_fn = cls._build_batched_additive_mask
    orig_flash_fn = cls._attend_flash_batch
    orig_scatter_fn = cls._scatter_plan_output
    orig_paged_fn = backend._attend_plan_bucket_paged
    orig_idx_fn = cls._build_paged_token_index

    backend._gather_plan_qkv = timed("gather", orig_gather)
    cls._build_batched_additive_mask = staticmethod(timed("mask", orig_mask_fn))
    cls._attend_flash_batch = staticmethod(timed("launch", orig_flash_fn))
    cls._scatter_plan_output = staticmethod(timed("scatter", orig_scatter_fn))
    # The paged/varlen path replaces gather+pack+scatter with a token index and
    # the kernel itself: ``paged`` is its top-level phase and ``idx`` the nested
    # index build (charged once, like mask and pack).
    # ``_attend_plan_bucket_paged`` is an instance method (it reads ``self``
    # state), so it cannot go through ``staticmethod`` like the module-level
    # helpers above: the call site is ``self._attend_plan_bucket_paged(...)``
    # with keyword args only, and an unbound wrapper drops ``self``.  Bind the
    # wrapper to this instance instead, exactly like ``_gather_plan_qkv`` above.
    # ``_build_paged_token_index`` really is a staticmethod, so it stays on the
    # class.
    backend._attend_plan_bucket_paged = timed("paged", orig_paged_fn)
    cls._build_paged_token_index = staticmethod(timed("idx", orig_idx_fn))

    # The request planner reads ``int(seq_lens[i])`` / ``int(req_pool_indices[i])``
    # once per request.  Each ``int()`` on a device tensor is a device->host
    # scalar read, i.e. a full sync -- ~2 per request.  It runs outside every
    # other timed helper, which is why it used to land in "unaccounted".
    attn_mod._plan_attention_requests = timed(
        "plan", attn_mod._plan_attention_requests
    )

    # Only the device packer exists now: the host-side ``_pack_flash_batch`` was
    # removed once the kernel was built in the model's own dtype (bf16), which
    # made the conversion it performed unnecessary.
    attn_mod._pack_flash_batch_device = timed(
        "pack", attn_mod._pack_flash_batch_device
    )

    def run_attn():
        phases.clear()
        calls.clear()
        with torch.no_grad(), forward_context(ctx):
            return attn.attn(q, k, v, fwd, save_kv_cache=True)

    for _ in range(WARMUP):
        run_attn()
    SYNC()

    # Arm profiling after warmup, once the kernel is cached. Reach into
    # the flash kernel cache (populated by the warmup) and enable_profile
    # on the single cached kernel for this (d, scale, dtype) key.
    if args.profile:
        from sglang.srt.hardware_backend.tpu.attention import (
            _FLASH_KERNELS,
            _PAGED_FLASH_KERNELS,
        )

        # Exactly one kernel of each kind is cached for this (d, scale, dtype).
        for name, cache in (("gathered", _FLASH_KERNELS), ("paged", _PAGED_FLASH_KERNELS)):
            if cache:
                next(iter(cache.values())).adapter.enable_profile()
                print(f"[ph] profiling armed for {name} {list(cache.keys())[0]}", flush=True)

    totals = []
    for _ in range(ROUNDS):
        SYNC()
        t0 = time.perf_counter()
        run_attn()
        SYNC()
        totals.append(time.perf_counter() - t0)

    if args.profile:
        print(f"\n[ph] collecting profile...", flush=True)
        if _FLASH_KERNELS:
            next(iter(_FLASH_KERNELS.values())).adapter.collect_profile(
                verbose=args.verbose
            )
        if _PAGED_FLASH_KERNELS:
            next(iter(_PAGED_FLASH_KERNELS.values())).adapter.collect_profile(
                verbose=args.verbose
            )


    total = statistics.median(totals)
    print(f"\n[ph] attn op total: {total*1e6:.0f} us  (median of {ROUNDS})")
    if args.raw:
        return

    def sum_of(name: str) -> float:
        return sum(phases.get(name, []))

    # Exclusive times: mask/pack/idx are charged once, inside launch/paged.
    plan = sum_of("plan")
    gather = sum_of("gather")
    launch = sum_of("launch")   # excludes mask and pack, which are nested
    paged = sum_of("paged")     # excludes mask and idx, which are nested
    mask = sum_of("mask")
    pack = sum_of("pack")
    idx = sum_of("idx")
    scatter = sum_of("scatter")
    acc = plan + gather + launch + paged + mask + pack + idx + scatter

    print(f"\n{'phase':<12}{'sum(us)':>10}{'calls':>7}{'%of total':>11}")
    print("-" * 40)
    for name in ("plan", "gather", "launch", "paged", "scatter"):
        s = sum_of(name)
        print(f"{name:<12}{s*1e6:>10.0f}{calls.get(name, 0):>7}{100*s/total:>11.1f}")
    print(f"{'  mask':<12}{mask*1e6:>10.0f}{calls.get('mask', 0):>7}"
          f"{100*mask/total:>11.1f}   (inside launch/paged)")
    print(f"{'  pack':<12}{pack*1e6:>10.0f}{calls.get('pack', 0):>7}"
          f"{100*pack/total:>11.1f}   (inside launch)")
    print(f"{'  idx':<12}{idx*1e6:>10.0f}{calls.get('idx', 0):>7}"
          f"{100*idx/total:>11.1f}   (inside paged)")
    print("-" * 40)
    print(f"{'SUM':<12}{acc*1e6:>10.0f}{'':>7}{100*acc/total:>11.1f}")
    print(f"{'unaccounted':<12}{(total-acc)*1e6:>10.0f}{'':>7}"
          f"{100*(total-acc)/total:>11.1f}")
    # ``launch``/``paged`` already exclude the nested mask/pack/idx phases, so
    # what remains of them is the kernel call itself (plus ctypes dispatch and
    # the device sync the launch performs).
    kernel = launch + paged
    print(f"\n[ph] kernel-only estimate: {kernel*1e6:.0f} us "
          f"of {total*1e6:.0f} us ({100*kernel/total:.1f}%)")
    # Every top-level timed call pays one SYNC() in _enter() *before* its window
    # opens, so that sync is charged to the e2e total but to no phase. That, not
    # missing work, is what "unaccounted" mostly is: the price of the
    # measurement itself.
    top_calls = sum(calls.get(n, 0) for n in ("plan", "gather", "launch", "scatter"))
    if top_calls:
        per_sync = (total - acc) / top_calls
        print(f"[ph] unaccounted / {top_calls} top-level calls = "
              f"{per_sync*1e6:.0f} us each: one profiler SYNC() per call, outside "
              f"the measured window.\n[ph] The phases above serialize work the "
              f"real run pipelines, so this total is not the op cost -- re-run "
              f"with --raw for that.")


if __name__ == "__main__":
    main()
