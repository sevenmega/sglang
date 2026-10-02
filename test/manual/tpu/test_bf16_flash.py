"""Does the tilelang flash-attention kernel stay numerically correct in bf16?

The codegen comment says ``tiu::fmm2`` takes "fp16 operands, fp32 accum", so
whether the Cube unit is correct with bf16 *operands* is an open question -- the
compile succeeding only proves the TIR/dtype plumbing accepts bf16, not that the
math is right.

Runs the same shapes as test_flash_attention.py in both fp16 (known good,
control) and bf16, against the same torch float32 reference.

    SGLANG_TPU_DEVICE_ID=6 python3 test_bf16_flash.py
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/workspace/tilelang")

from tilelang.tpu.kernels.attention import flash_attention_gqa

MASK_NEG = -30000.0
D = 128
SCALE = float(D) ** -0.5


def build_causal_mask(sq, skv):
    query_offset = skv - sq
    i = torch.arange(sq).unsqueeze(1)
    j = torch.arange(skv).unsqueeze(0)
    disallowed = j > (i + query_offset)
    mask = torch.zeros(sq, skv, dtype=torch.float32)
    mask.masked_fill_(disallowed, MASK_NEG)
    return mask


def ref_attention(q, k, v, mask, scale):
    b, hq, sq, d = q.shape
    _, hkv, skv, _ = k.shape
    head_rep = hq // hkv
    out = torch.empty(b, hq, sq, d, dtype=torch.float32)
    for bi in range(b):
        for h in range(hq):
            kv_h = h // head_rep
            qi = q[bi, h, :, :].float()
            ki = k[bi, kv_h, :, :].float()
            vi = v[bi, kv_h, :, :].float()
            scores = (qi @ ki.t()) * scale + mask[bi]
            p = torch.softmax(scores, dim=-1)
            out[bi, h, :, :] = p @ vi
    return out


def run_case(label, dtype, B, SQ, SKV, HQ, HKV):
    kernel = flash_attention_gqa(d=D, sm_scale=SCALE, dtype=dtype_str(dtype))
    torch.manual_seed(0)
    q = torch.randn(B, HQ, SQ, D, dtype=dtype)
    k = torch.randn(B, HKV, SKV, D, dtype=dtype)
    v = torch.randn(B, HKV, SKV, D, dtype=dtype)
    mask2d = build_causal_mask(SQ, SKV)
    mask = mask2d.unsqueeze(0).expand(B, SQ, SKV).contiguous()

    out = kernel(q, k, v, mask).float()
    ref = ref_attention(q, k, v, mask, SCALE)

    diff = (out - ref).abs()
    rel = (diff.norm() / ref.norm().clamp_min(1e-6)).item()
    cos = torch.nn.functional.cosine_similarity(
        out.flatten(), ref.flatten(), dim=0
    ).item()
    nan = bool(torch.isnan(out).any())
    ok = (not nan) and cos > 0.999 and rel < 0.02
    print(
        f"  [{label:>5}] B={B} Sq={SQ} Skv={SKV}  "
        f"maxabs={diff.max().item():.6f}  rel={rel:.4%}  "
        f"cos={cos:.6f}  nan={nan}  {'PASS' if ok else 'FAIL'}"
    )
    return ok


def dtype_str(dt):
    return "bfloat16" if dt == torch.bfloat16 else "float16"


def main():
    shapes = [
        (1, 8, 128, 16, 8),
        (32, 256, 256, 16, 8),   # the prefill bucket at issue
        (1, 1, 384, 16, 8),
    ]
    print("=== fp16 (control: known-good) ===")
    fp16_ok = all(
        run_case("fp16", torch.float16, *s) for s in shapes
    )
    print("\n=== bf16 (under test) ===")
    bf16_ok = all(
        run_case("bf16", torch.bfloat16, *s) for s in shapes
    )
    print()
    print(f"fp16 all-pass: {fp16_ok}")
    print(f"bf16 all-pass: {bf16_ok}")
    if not bf16_ok:
        print("\nVERDICT: do NOT switch the kernel to bf16.")
        sys.exit(1)
    print("\nVERDICT: bf16 is numerically sound; safe to wire into sglang.")


if __name__ == "__main__":
    main()
