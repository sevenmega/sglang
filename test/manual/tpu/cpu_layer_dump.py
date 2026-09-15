"""CPU golden reference: dump one Qwen3 decoder layer's inputs + output.

Runs a single prefill forward of Qwen3-0.6B on CPU in bf16 (HF transformers,
eager attention) and captures, via forward hooks on ``model.model.layers[N]``,
everything that layer receives and produces:

    inputs : hidden_states, attention_mask, position_ids, position_embeddings
    output : hidden_states (the layer's returned tensor)

These are saved to a .pt file to serve as the golden reference for the TPU
single-layer replay test (tpu_layer_test.py). Layer index and prompt are
configurable via env (LAYER, PROMPT); defaults: layer 0, capital-of-France.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL_PATH", "/workspace/Qwen3-0.6B")
LAYER = int(os.environ.get("LAYER", "0"))
PROMPT = os.environ.get("PROMPT", "The capital of France is")
OUT = os.environ.get("REF_OUT", "/tmp/layer_ref.pt")

torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager"
)
model.eval()

layer = model.model.layers[LAYER]

captured = {}


def pre_hook(module, args, kwargs):
    # hidden_states is passed positionally; the rest as kwargs.
    captured["hidden_states"] = args[0].detach().clone()
    captured["attention_mask"] = (
        kwargs["attention_mask"].detach().clone()
        if kwargs.get("attention_mask") is not None
        else None
    )
    captured["position_ids"] = (
        kwargs["position_ids"].detach().clone()
        if kwargs.get("position_ids") is not None
        else None
    )
    pe = kwargs.get("position_embeddings")
    captured["position_embeddings"] = (
        (pe[0].detach().clone(), pe[1].detach().clone()) if pe is not None else None
    )
    # Record what keys/positional args actually arrived, for debugging.
    captured["_arg_count"] = len(args)
    captured["_kwarg_keys"] = sorted(kwargs.keys())


def post_hook(module, args, kwargs, output):
    out = output[0] if isinstance(output, tuple) else output
    captured["output"] = out.detach().clone()


h1 = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
h2 = layer.register_forward_hook(post_hook, with_kwargs=True)

enc = tok(PROMPT, return_tensors="pt")
with torch.no_grad():
    model(**enc)

h1.remove()
h2.remove()


def _spec(t):
    if t is None:
        return "None"
    return f"{tuple(t.shape)}:{str(t.dtype).replace('torch.', '')}"


print(f"=== layer {LAYER} capture (prompt={PROMPT!r}) ===")
print("  positional args to layer:", captured["_arg_count"])
print("  kwarg keys to layer:", captured["_kwarg_keys"])
print("  hidden_states  :", _spec(captured["hidden_states"]))
print("  attention_mask :", _spec(captured["attention_mask"]))
print("  position_ids   :", _spec(captured["position_ids"]))
pe = captured["position_embeddings"]
print("  position_emb   :", (_spec(pe[0]), _spec(pe[1])) if pe else "None")
print("  output         :", _spec(captured["output"]))

meta = {
    "model": MODEL,
    "layer": LAYER,
    "prompt": PROMPT,
    "input_ids": enc["input_ids"],
    "hidden_states": captured["hidden_states"],
    "attention_mask": captured["attention_mask"],
    "position_ids": captured["position_ids"],
    "position_embeddings": captured["position_embeddings"],
    "output": captured["output"],
}
torch.save(meta, OUT)
print(f"\nsaved reference -> {OUT}")
