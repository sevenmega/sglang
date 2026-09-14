import os
import sglang as sgl
from transformers import AutoTokenizer


def main():
    tok = AutoTokenizer.from_pretrained("/workspace/Qwen3-0.6B")
    base = tok("The capital of France is").input_ids
    # + " Paris. The capital of"
    prefix = base + [12095, 13, 576, 6722, 315]
    print("prefix ids:", prefix, flush=True)

    engine = sgl.Engine(
        model_path="/workspace/Qwen3-0.6B", tp_size=1, mem_fraction_static=0.6,
        max_running_requests=1, context_length=2048,
        base_gpu_id=int(os.environ.get("SGLANG_TPU_DEVICE_ID", "0")),
    )
    out = engine.generate(
        input_ids=prefix,
        sampling_params={"temperature": 0.0, "max_new_tokens": 1},
        return_logprob=True,
        token_ids_logprob=[15344, 9625],  # Italy vs France
    )
    ml = out["meta_info"]
    print("TPU chosen:", ml["output_token_logprobs"], flush=True)
    for lp, tid, _ in ml["output_token_ids_logprobs"][0]:
        name = {15344: "Italy", 9625: "France"}[tid]
        print(f"    id={tid} ({name}) lp={lp:.4f}", flush=True)
    engine.shutdown()
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
