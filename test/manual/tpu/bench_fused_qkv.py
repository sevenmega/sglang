"""A/B the fused qkv/norm/RoPE/cache-write path against the torch chain.

One decode step and one prefill pass per repetition, wall clock around
``engine.generate`` (the engine is synchronous).  Prefill is measured with
``max_new_tokens=1`` and decode with a long enough generation that the prefill
cost is amortised; the split matters because the fused kernel replaces one
per-layer chain of ~15 launches, which is a per-*token* cost in both phases.

Run twice and compare (the flag is read once, at engine construction):

    SGLANG_TPU_USE_FUSED_QKV_ROPE=0 python bench_fused_qkv.py --input-len 512
    SGLANG_TPU_USE_FUSED_QKV_ROPE=1 python bench_fused_qkv.py --input-len 512

The first run with the flag on pays a ~1 min kernel build per query-row tile
(one for prefill-sized batches, one for short decode batches) during warmup, so
the warmup rounds are discarded.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import sglang as sgl

WARMUP = 2


def _prompt(input_len: int) -> str:
    """A prompt of roughly ``input_len`` tokens (the tokenizer is not exact)."""
    filler = " the capital of France is Paris and the river runs through it,"
    return "Tell me a long story." + filler * (input_len // 12)


def _time_generate(engine, prompt, max_new_tokens, rounds):
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        out = engine.generate(
            prompt,
            {"temperature": 0.0, "max_new_tokens": max_new_tokens,
             "ignore_eos": True},
        )
        times.append(time.perf_counter() - t0)
    n_out = out["meta_info"]["completion_tokens"]
    return statistics.median(times), n_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/Qwen3-0.6B")
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    flag = os.environ.get("SGLANG_TPU_USE_FUSED_QKV_ROPE", "0")
    engine = sgl.Engine(
        model_path=args.model_path,
        tp_size=1,
        mem_fraction_static=0.6,
        max_running_requests=1,
        context_length=2048,
        base_gpu_id=int(os.environ.get("SGLANG_TPU_DEVICE_ID", "0")),
    )
    prompt = _prompt(args.input_len)

    try:
        for _ in range(WARMUP):
            _time_generate(engine, prompt, args.decode_tokens, 1)

        prefill, _ = _time_generate(engine, prompt, 1, args.rounds)
        decode, n_out = _time_generate(engine, prompt, args.decode_tokens, args.rounds)
    finally:
        engine.shutdown()

    # One prefill + (n_out - 1) decode steps, so the decode step cost is
    # (total - prefill) / (n_out - 1) -- the prefill is the same prompt each time.
    step = (decode - prefill) / max(1, n_out - 1)
    print(f"fused_qkv_rope={flag}  input_len~{args.input_len}")
    print(f"  prefill (max_new=1)   : {prefill * 1e3:8.1f} ms")
    print(f"  decode ({n_out} tok)     : {decode * 1e3:8.1f} ms")
    print(f"  per decode step       : {step * 1e3:8.2f} ms")


if __name__ == "__main__":
    main()
