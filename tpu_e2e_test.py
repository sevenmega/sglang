import sglang as sgl

def main():
    engine = sgl.Engine(
        model_path="/workspace/Qwen3-0.6B",
        tp_size=1,
        base_gpu_id=int(__import__("os").environ.get("SGLANG_TPU_DEVICE_ID", "0")),
        mem_fraction_static=0.6,
        max_running_requests=1,
        context_length=2048,
        log_level="info",
        skip_tokenizer_init=False,
    )
    prompts = ["The capital of France is"]
    out = engine.generate(
        prompts,
        {"temperature": 0.0, "max_new_tokens": 16},
    )
    for p, o in zip(prompts, out):
        print("PROMPT:", repr(p))
        print("OUTPUT:", repr(o["text"]))
    engine.shutdown()

if __name__ == "__main__":
    main()
