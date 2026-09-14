import sglang as sgl


def main():
    engine = sgl.Engine(
        model_path="/workspace/Qwen3-0.6B",
        tp_size=1,
        mem_fraction_static=0.6,
        max_running_requests=1,
        context_length=2048,
        base_gpu_id=int(__import__("os").environ.get("SGLANG_TPU_DEVICE_ID", "0")),
    )
    for trial in range(2):
        out = engine.generate(
            "The capital of France is",
            {"temperature": 0.0, "max_new_tokens": 16},
            return_logprob=True,
        )
        ids = [t[1] for t in out["meta_info"]["output_token_logprobs"]]
        print(f"trial{trial} TEXT: {out['text']!r}", flush=True)
        print(f"trial{trial} IDS : {ids}", flush=True)
    engine.shutdown()
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
