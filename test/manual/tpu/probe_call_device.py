"""Break down a TPU flash-attention launch: where does the 'launch' time go?

Times the public launch call and the allocator separately, for the exact prefill
bucket shape (B=32, Hq=16, Hkv=8, D=128, Sq=256, Skv=256):

  launch -- kernel(q, k, v, mask): dims resolve + output alloc + ctypes dispatch
            + the post-launch py_sync_device, because the launch is *synchronous*
            (TPUKernel._launch_device syncs right after run_device_ptrs)
  alloc  -- a bare torch.empty of the output, to isolate allocator cost

`kernel(...)` is the device-pointer path -- the same call sglang's
``_attend_flash_batch`` makes -- so there is no separate ``call_device`` to
mirror, and no host marshalling inside it to account for.

Run:
    SGLANG_TPU_DEVICE_ID=6 python3 probe_call_device.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "/workspace/tilelang")

WARMUP = 3
ROUNDS = 10


def main():
    from tilelang.tpu.ppl_runner import ensure_tpu_device

    # torch_tpu selects the chip; it ignores TPU_VISIBLE_DEVICES. The PPL
    # runtime maps that logical id to its own physical id internally
    # (get_tpu_device), so nothing here needs to know the mapping.
    dev = f"tpu:{ensure_tpu_device()}"

    from sglang.srt.hardware_backend.tpu.attention import _get_flash_kernel

    B, Hq, Hkv, D = 32, 16, 8, 128
    Sq, Skv = 256, 256
    scaling = D ** -0.5
    dtype = torch.bfloat16

    kernel = _get_flash_kernel(
        hq=Hq, sq=Sq, skv=Skv, d=D, hkv=Hkv, scaling=scaling, dtype=dtype
    )
    print(f"[cd] device={dev} kernel type={type(kernel).__name__}")

    # The packed device inputs _attend_flash_batch hands to the kernel.
    q_b = torch.randn(B, Hq, Sq, D, dtype=dtype, device=dev)
    k_b = torch.randn(B, Hkv, Skv, D, dtype=dtype, device=dev)
    v_b = torch.randn(B, Hkv, Skv, D, dtype=dtype, device=dev)
    mask = torch.zeros(B, Sq, Skv, dtype=torch.float32, device=dev)
    inputs = (q_b, k_b, v_b, mask)

    sy = torch.get_device_module(dev).synchronize

    # The kernel call blocks until the device finishes, so it is measured as a
    # single number; splitting dispatch from completion would be fiction.
    times = {"launch": []}

    def once():
        sy()
        t0 = time.perf_counter()
        out = kernel(*inputs)  # synchronous: returns after py_sync_device
        times["launch"].append(time.perf_counter() - t0)
        return out

    for _ in range(WARMUP):
        once()
    for _ in range(ROUNDS):
        once()

    print(f"\n[cd] launch (median of {ROUNDS}), B={B}")
    print("-" * 30)
    m = statistics.median(times["launch"])
    print(f"{'launch':<10}{m*1e3:>10.2f} ms")
    print("-" * 30)

    # The output alloc happens inside the launch; time a bare one to size it.
    sy()
    t0 = time.perf_counter()
    for _ in range(ROUNDS):
        _ = torch.empty(B, Hq, Sq, D, dtype=dtype, device=dev)
    sy()
    print(f"\n[cd] bare torch.empty([{B},{Hq},{Sq},{D}]) x{ROUNDS}: "
          f"{(time.perf_counter()-t0)/ROUNDS*1e3:.2f} ms each")


if __name__ == "__main__":
    main()
