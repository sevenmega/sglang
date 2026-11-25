"""Time the driver-side pieces of the paged attention path, model-free.

``profile_attn_phases.py`` can only see phases the driver exposes as methods.
The two biggest remaining costs on the paged path -- building the gather index
and copying each request's real output rows out of the per-request padded slab
-- live inline in ``_attend_plan_bucket_paged``, so they show up as "paged
exclusive" together with the kernel call.  This probe times them directly, on
their own, with the same shapes a B=32 / 256-token prefill bucket uses.

Run:
    SGLANG_TPU_DEVICE_ID=7 python3 test/manual/tpu/probe_paged_overhead.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import torch

BATCH = 32
Q_LEN = 256
KV_LEN = 256
HQ, HKV, D = 16, 8, 128
ROWS = 915_000
ROUNDS = 5


def main() -> None:
    dev_id = int(os.environ.get("SGLANG_TPU_DEVICE_ID", "0"))
    import torch_tpu  # noqa: F401

    torch.tpu.set_device(dev_id)
    dev = f"tpu:{dev_id}"
    sync = torch.get_device_module(dev).synchronize

    from sglang.srt.hardware_backend.tpu.attention import (
        TpuAttnBackend,
        _AttnReqPlan,
        _flash_pad_lens,
        _get_flash_kernel,
        _get_paged_flash_kernel,
        _pack_flash_batch_device,
    )

    q_pad, kv_pad = _flash_pad_lens(Q_LEN, KV_LEN)
    plans = [
        _AttnReqPlan(
            q_start=b * Q_LEN,
            q_len=Q_LEN,
            kv_start=0,
            kv_len=KV_LEN,
            req_pool_idx=b,
        )
        for b in range(BATCH)
    ]
    q_lens = [p.q_len for p in plans]
    kv_lens = [p.kv_len for p in plans]

    req_to_token = torch.randint(
        0, ROWS, (BATCH, kv_pad), dtype=torch.int32, device=dev
    )
    # Kernel output as the driver sees it, and the packed output it scatters into.
    slab = torch.empty((BATCH, q_pad, HQ, D), dtype=torch.bfloat16, device=dev)
    output = torch.empty((BATCH * Q_LEN, HQ, D), dtype=torch.bfloat16, device=dev)
    # Deterministic fill, not ``empty``: the kernel averages V, so uninitialized
    # bits (which can encode NaN) would make every output NaN and the
    # repeat-invariance check below meaningless.
    k_cache = torch.ones((ROWS, HKV, D), dtype=torch.bfloat16, device=dev)
    v_cache = torch.ones_like(k_cache)

    def build_idx():
        return TpuAttnBackend._build_paged_token_index(
            plans=plans, req_to_token=req_to_token, n_kv_heads=HKV, kv_pad=kv_pad
        )

    def build_mask():
        return TpuAttnBackend._build_batched_additive_mask(
            q_lens=q_lens,
            kv_lens=kv_lens,
            q_pad=q_pad,
            kv_pad=kv_pad,
            causal=True,
            sliding_window_size=None,
            device=dev,
        ).contiguous()

    def build_q_off():
        return torch.tensor(
            [p.q_start for p in plans], dtype=torch.int32, device=dev
        )

    def copy_out():
        for b, plan in enumerate(plans):
            output[plan.q_start : plan.q_start + plan.q_len] = slab[b, : plan.q_len]

    def gather_legacy():
        # What the gathered path pays per request: pull its K/V rows out of the
        # cache (the kernel then reads dense tiles).
        for plan in plans:
            tokens = req_to_token[
                plan.req_pool_idx, plan.kv_start : plan.kv_start + plan.kv_len
            ]
            k_cache[tokens]
            v_cache[tokens]

    # The gathered path's other half: pack the per-request [H, S, D] views into
    # dense [B, H, s_pad, D] batches (q, k and v).
    q_items = [
        torch.randn((HQ, Q_LEN, D), dtype=torch.bfloat16, device=dev)
        for _ in range(BATCH)
    ]
    kv_items = [
        torch.randn((HKV, KV_LEN, D), dtype=torch.bfloat16, device=dev)
        for _ in range(BATCH)
    ]

    def pack_legacy():
        for items, heads in ((q_items, HQ), (kv_items, HKV), (kv_items, HKV)):
            _pack_flash_batch_device(
                items=items, n_heads=heads, s_pad=Q_LEN, dtype=torch.bfloat16
            )

    scaling = float(D) ** -0.5
    dtype = torch.bfloat16

    # Packed inputs, the shape the driver hands each kernel.
    q_packed = torch.randn((BATCH * Q_LEN, HQ, D), dtype=dtype, device=dev)
    mask = build_mask()
    idx = build_idx()
    q_off = build_q_off()
    k_dense = torch.randn((BATCH, HKV, Q_LEN, D), dtype=dtype, device=dev)
    v_dense = torch.randn((BATCH, HKV, Q_LEN, D), dtype=dtype, device=dev)
    q_dense = torch.randn((BATCH, HQ, Q_LEN, D), dtype=dtype, device=dev)

    def launch_gathered():
        gathered_kern(q_dense, k_dense, v_dense, mask)

    def launch_paged():
        paged_kern(q_packed, k_cache, v_cache, idx, mask, q_off)

    def bench(name, fn):
        for _ in range(2):
            fn()
        sync()
        times = []
        for _ in range(ROUNDS):
            sync()
            t0 = time.perf_counter()
            fn()
            sync()
            times.append(time.perf_counter() - t0)
        ms = statistics.median(times) * 1e3
        print(f"  {name:<16}{ms:>8.2f} ms")
        return ms

    print(f"[probe] B={BATCH} q={Q_LEN} kv={KV_LEN} q_pad={q_pad} kv_pad={kv_pad}")
    total = 0.0
    total += bench("idx", build_idx)
    total += bench("mask", build_mask)
    total += bench("q_off", build_q_off)
    total += bench("copy_out", copy_out)
    total += bench("gather(legacy)", gather_legacy)
    total += bench("pack(legacy)", pack_legacy)
    print(f"  {'driver sum':<16}{total:>8.2f} ms")

    # Coexistence A/B: time the paged kernel alone, then again after the
    # gathered kernel has been built and launched in the same process. A large
    # regression (and/or changed output) means the two kernels do not share a
    # process cleanly.
    paged_kern = _get_paged_flash_kernel(d=D, scaling=scaling, dtype=dtype)
    bench("paged alone", launch_paged)
    out_before = paged_kern(q_packed, k_cache, v_cache, idx, mask, q_off).clone()
    gathered_kern = _get_flash_kernel(
        hq=HQ, sq=Q_LEN, skv=KV_LEN, d=D, hkv=HKV, scaling=scaling, dtype=dtype
    )
    bench("gathered", launch_gathered)
    bench("paged after", launch_paged)
    out_after = paged_kern(q_packed, k_cache, v_cache, idx, mask, q_off).clone()
    sync()
    delta = (out_before.float() - out_after.float()).abs().max().item()
    print(f"  paged output drift after gathered built: {delta:.6g}")


if __name__ == "__main__":
    sys.exit(main())
