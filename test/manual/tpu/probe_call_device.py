"""Break down TPUKernel.call_device: where do the ~90ms of 'launch' go?

The attn phase profiler attributes 90ms to 'launch' (_attend_flash_batch) but
bigTpuProfile says the kernel runs ~32ms on-device. This probe inlines
call_device's steps and times each one, for the exact prefill bucket shape
(B=32, Hq=16, Hkv=8, D=128, Sq=256, Skv=256):

  dims    -- resolve dynamic dims from input shapes (pure python)
  alloc   -- torch.empty output on the TPU device
  dispatch-- rt.run_device_ptrs (ctypes launch, async)
  sync    -- rt.lib.py_sync_device (blocks until kernel completes)

Run:
    SGLANG_TPU_DEVICE_ID=6 python3 probe_call_device.py
"""

from __future__ import annotations

import statistics
import sys
import time

import torch

sys.path.insert(0, "/workspace/tilelang")

WARMUP = 3
ROUNDS = 10


def main():
    import torch_tpu

    torch_tpu.tpu.set_device(6)
    dev = "tpu:6"

    from tilelang.tpu.ppl_runner import _TORCH_DTYPE, _concrete_dim
    from sglang.srt.hardware_backend.tpu.attention import (
        _get_flash_kernel,
        _sync_rt_device,
    )

    B, Hq, Hkv, D = 32, 16, 8, 128
    Sq, Skv = 256, 256
    scaling = D ** -0.5
    dtype = torch.bfloat16

    kernel = _get_flash_kernel(
        hq=Hq, sq=Sq, skv=Skv, d=D, hkv=Hkv, scaling=scaling, dtype=dtype
    )
    print(f"[cd] kernel type={type(kernel).__name__}")
    # _get_flash_kernel returns a JITKernel; the TPUKernel (with call_device /
    # _ensure_runtime) is behind the adapter.
    tpu_kernel = kernel.adapter._tpu_kernel
    print(f"[cd] tpu_kernel type={type(tpu_kernel).__name__}")

    # Build the packed device inputs once (these are what _attend_flash_batch
    # hands to call_device).
    q_b = torch.randn(B, Hq, Sq, D, dtype=dtype, device=dev)
    k_b = torch.randn(B, Hkv, Skv, D, dtype=dtype, device=dev)
    v_b = torch.randn(B, Hkv, Skv, D, dtype=dtype, device=dev)
    mask = torch.zeros(B, Sq, Skv, dtype=torch.float32, device=dev)
    inputs = (q_b, k_b, v_b, mask)

    rt = tpu_kernel._ensure_runtime()
    info = tpu_kernel.info
    SYNC = lambda: torch.get_device_module(dev).synchronize()

    def resolve_dims():
        dims_by_name = {}
        for arg, t in zip(info.inputs, inputs):
            for i, s in enumerate(arg.shape):
                name = str(s)
                if name not in info.dyn_dims:
                    continue
                dims_by_name[name] = int(t.shape[i])
        return dims_by_name

    dims = resolve_dims()
    tpu_device = f"tpu:{tpu_kernel.device}"

    # Timing helpers -- each step bracketed by a device sync so we measure the
    # serialized cost of that step alone.
    times = {"dims": [], "alloc": [], "dispatch": [], "sync": [], "total": []}

    def once():
        _sync_rt_device()
        SYNC()
        t0 = time.perf_counter()

        d = resolve_dims()
        t1 = time.perf_counter()

        outs = []
        for arg in info.outputs:
            shape = tuple(_concrete_dim(s, d) for s in arg.shape)
            odt = _TORCH_DTYPE[arg.torch_dtype]
            outs.append(torch.empty(shape, dtype=odt, device=tpu_device))
        SYNC()
        t2 = time.perf_counter()

        in_addrs = [int(t.data_ptr()) for t in inputs]
        out_addrs = [int(t.data_ptr()) for t in outs]
        rt.run_device_ptrs(in_addrs, d, out_addrs)   # async dispatch
        t3 = time.perf_counter()

        rt.lib.py_sync_device(rt.handle)             # blocks on completion
        t4 = time.perf_counter()

        times["dims"].append(t1 - t0)
        times["alloc"].append(t2 - t1)
        times["dispatch"].append(t3 - t2)
        times["sync"].append(t4 - t3)
        times["total"].append(t4 - t0)
        return outs[0]

    for _ in range(WARMUP):
        once()
    for _ in range(ROUNDS):
        once()

    print(f"\n[cd] call_device internal breakdown (median of {ROUNDS}), B={B}")
    print(f"{'step':<10}{'ms':>10}{'% total':>10}")
    print("-" * 30)
    tot = statistics.median(times["total"])
    for k in ["dims", "alloc", "dispatch", "sync"]:
        m = statistics.median(times[k])
        print(f"{k:<10}{m*1e3:>10.2f}{100*m/tot:>10.1f}")
    print("-" * 30)
    print(f"{'total':<10}{tot*1e3:>10.2f}{100:>10.1f}")

    # Also time a bare torch.empty on device to isolate allocator cost.
    SYNC()
    t0 = time.perf_counter()
    for _ in range(ROUNDS):
        _ = torch.empty(B, Hq, Sq, D, dtype=dtype, device=dev)
    SYNC()
    print(f"\n[cd] bare torch.empty([{B},{Hq},{Sq},{D}]) x{ROUNDS}: "
          f"{(time.perf_counter()-t0)/ROUNDS*1e3:.2f} ms each")


if __name__ == "__main__":
    main()
