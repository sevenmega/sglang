"""Regression guard: the on-device attention mask must equal the CPU mask.

``_build_additive_mask`` / ``_build_batched_additive_mask`` moved from host
assembly + upload to on-device construction in 1c6e46a47 ("update with device
tensor support").  On that build of torch_tpu a *broadcasting* comparison swaps
its operands -- ``[q,1] > [1,kv]`` evaluates as ``[1,kv] > [q,1]`` -- so the
causal mask came out transposed for square shapes and as unrelated garbage for
``q_len != kv_len``.  Every attention path consumed it (flash kernel or the
matmul fallback), which is why ``tpu_lp16.py`` produced a fluent but unrelated
continuation.  ``+``/``*`` broadcast correctly; comparisons do not, so the
builder now ``.expand()``s both operands first.

This compares the device-built mask against the same build on CPU for the
shapes the engine actually uses.  A mask that is transposed, mirrored or
otherwise shifted fails here; the flash tests cannot catch it, because they
build their reference from the same mask.

Run:
    SGLANG_TPU_DEVICE_ID=7 python3 test/manual/tpu/probe_mask_device.py
"""

from __future__ import annotations

import os

import torch


def main() -> None:
    dev_id = int(os.environ.get("SGLANG_TPU_DEVICE_ID", "0"))
    import torch_tpu  # noqa: F401

    torch.tpu.set_device(dev_id)
    dev = f"tpu:{dev_id}"

    from sglang.srt.hardware_backend.tpu.attention import TpuAttnBackend

    ok = True
    for q_len, kv_len in ((5, 5), (5, 8), (1, 6), (64, 64), (64, 256)):
        cpu = TpuAttnBackend._build_additive_mask(
            q_len=q_len,
            kv_len=kv_len,
            causal=True,
            sliding_window_size=None,
            device="cpu",
        )
        tpu = TpuAttnBackend._build_additive_mask(
            q_len=q_len,
            kv_len=kv_len,
            causal=True,
            sliding_window_size=None,
            device=dev,
        )
        got = tpu.float().cpu()
        same = torch.equal(cpu, got)
        ok &= same
        print(f"single q={q_len} kv={kv_len}: {'OK' if same else 'MISMATCH'}")
        if not same:
            print("  cpu:\n", cpu[:4, :8])
            print("  tpu:\n", got[:4, :8])
            print("  max abs diff:", (cpu - got).abs().max().item())

    q_lens, kv_lens, q_pad, kv_pad = [5, 3], [5, 8], 8, 8
    cpu = TpuAttnBackend._build_batched_additive_mask(
        q_lens=q_lens,
        kv_lens=kv_lens,
        q_pad=q_pad,
        kv_pad=kv_pad,
        causal=True,
        sliding_window_size=None,
        device="cpu",
    )
    tpu = TpuAttnBackend._build_batched_additive_mask(
        q_lens=q_lens,
        kv_lens=kv_lens,
        q_pad=q_pad,
        kv_pad=kv_pad,
        causal=True,
        sliding_window_size=None,
        device=dev,
    )
    got = tpu.float().cpu()
    same = torch.equal(cpu, got)
    ok &= same
    print(f"batched: {'OK' if same else 'MISMATCH'}")
    if not same:
        bad = (cpu != got).nonzero()
        print("  first mismatches:", bad[:10].tolist())
        for b, i, j in bad[:3].tolist():
            print(f"  [{b},{i},{j}] cpu={cpu[b, i, j].item()} tpu={got[b, i, j].item()}")

    print("EXIT", 0 if ok else 1)


if __name__ == "__main__":
    main()
