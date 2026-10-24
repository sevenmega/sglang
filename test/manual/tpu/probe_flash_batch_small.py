"""Numerics check for the batched flash driver at small (cached) shapes.

``tpu_lp16.py`` gives identical output for its two trials with
``SGLANG_TPU_USE_FLASH_ATTN=0`` but a different (still fluent) answer for the
second trial with flash on -- that trial sees the prompt as a radix-cache hit,
so it runs the tiny ``q_len=1`` / ``kv_len=5`` shape instead of the 5x5 prefill.
This drives ``TpuAttnBackend._attend_flash_batch`` (exactly what the engine
calls) at those shapes and compares against a float32 torch reference.

Run:
    SGLANG_TPU_DEVICE_ID=7 python3 test/manual/tpu/probe_flash_batch_small.py
"""

from __future__ import annotations

import math
import os

import torch

HQ = 16
HKV = 8
D = 128


def ref_attention(q, k, v, scaling, q_len, kv_len):
    """Float32 per-head reference on CPU: q [Hq,Sq,D], k/v [Hkv,Skv,D]."""
    q = q.float()
    k = k.float()
    v = v.float()
    out = torch.zeros(HQ, q_len, D)
    group = HQ // HKV
    offset = kv_len - q_len
    for h in range(HQ):
        s = q[h, :q_len] @ k[h // group, :kv_len].t() * scaling
        i = torch.arange(q_len).unsqueeze(1) + offset
        j = torch.arange(kv_len).unsqueeze(0)
        s = s.masked_fill(j > i, -30000.0)
        out[h] = (torch.softmax(s, dim=-1) @ v[h // group, :kv_len])
    return out


def main() -> None:
    dev_id = int(os.environ.get("SGLANG_TPU_DEVICE_ID", "0"))
    import torch_tpu  # noqa: F401

    torch.tpu.set_device(dev_id)
    dev = f"tpu:{dev_id}"

    from sglang.srt.hardware_backend.tpu.attention import (
        TpuAttnBackend,
        _FLASH_BLOCK_M,
        _FLASH_BLOCK_N,
    )

    torch.manual_seed(0)
    scaling = 1.0 / math.sqrt(D)

    ok = True
    for q_len, kv_len in ((5, 5), (1, 5), (1, 6), (1, 1), (2, 9)):
        q = torch.randn(HQ, q_len, D, dtype=torch.bfloat16)
        k = torch.randn(HKV, kv_len, D, dtype=torch.bfloat16)
        v = torch.randn(HKV, kv_len, D, dtype=torch.bfloat16)

        q_pad = -(-q_len // _FLASH_BLOCK_M) * _FLASH_BLOCK_M
        kv_pad = -(-kv_len // _FLASH_BLOCK_N) * _FLASH_BLOCK_N
        if kv_pad < q_pad:
            kv_pad = q_pad

        got = TpuAttnBackend._attend_flash_batch(
            qs=[q.to(dev)],
            ks=[k.to(dev)],
            vs=[v.to(dev)],
            q_lens=[q_len],
            kv_lens=[kv_len],
            q_pad=q_pad,
            kv_pad=kv_pad,
            scaling=scaling,
            causal=True,
            sliding_window_size=None,
        )
        if got is None:
            print(f"q={q_len} kv={kv_len}: LAUNCH FAILED")
            ok = False
            continue

        # [B, Hq, q_pad, D] -> [Hq, q_len, D]
        got_cpu = got[0, :, :q_len].float().cpu()
        want = ref_attention(q, k, v, scaling, q_len, kv_len)
        err = (got_cpu - want).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            got_cpu.flatten(), want.flatten(), dim=0
        ).item()
        good = err < 0.05 and cos > 0.999
        ok &= good
        print(
            f"q={q_len:<2} kv={kv_len:<3} q_pad={q_pad:<3} kv_pad={kv_pad:<4} "
            f"max_abs_err={err:.5f} cos={cos:.6f} {'OK' if good else 'MISMATCH'}"
        )
        if not good:
            print("    got[0,0,:6] :", got_cpu[0, 0, :6].tolist())
            print("    want[0,0,:6]:", want[0, 0, :6].tolist())

    print("EXIT", 0 if ok else 1)


if __name__ == "__main__":
    main()
