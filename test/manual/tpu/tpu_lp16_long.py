"""``tpu_lp16.py`` over a prompt long enough to look like real prefill.

``tpu_lp16.py`` prompts in 5 tokens, so its prefill is padding to exactly one
row tile and almost all of the run is decode.  This repeats a paragraph
instead: ``--repeat 180`` (the default) is 4861 tokens, a realistic prefill
inside ``context_length``, and long enough that the fused ops run as they would
in production (the o_proj fusion has no size gate).

Prints the prompt length, the generated ids and the first token's logprob, so
two configs can be compared on more than the ids -- those are not reproducible
run to run on this box, so run it several times per config and compare the
majority.  One process, one generate.
"""

import argparse
import os

import sglang as sgl

PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog while the morning train pulls "
    "out of the station and the city slowly wakes up around it. "
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=180, help="paragraph repeats")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()

    engine = sgl.Engine(
        model_path="/workspace/Qwen3-0.6B",
        tp_size=1,
        mem_fraction_static=0.6,
        max_running_requests=1,
        context_length=8192,
        base_gpu_id=int(os.environ.get("SGLANG_TPU_DEVICE_ID", "0")),
    )
    prompt = PARAGRAPH * args.repeat
    out = engine.generate(
        prompt,
        {"temperature": 0.0, "max_new_tokens": args.max_new_tokens},
        return_logprob=True,
    )
    # Printed so a run can be told apart from a short-prompt one.
    print(f"PROMPT TOKENS: {out['meta_info'].get('prompt_tokens')}", flush=True)
    lp = out["meta_info"]["output_token_logprobs"]
    ids = [t[1] for t in lp]
    print(f"IDS : {ids}", flush=True)
    print(f"FIRST LP: {lp[0][0]:.6f}", flush=True)
    print(f"TEXT: {out['text']!r}", flush=True)
    engine.shutdown()
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
