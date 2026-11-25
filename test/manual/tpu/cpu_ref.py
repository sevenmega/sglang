import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mp = "/workspace/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(mp)
model = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.bfloat16).eval()

prompt = "The capital of France is"
ids = tok(prompt, return_tensors="pt").input_ids
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=16, do_sample=False, temperature=None, top_p=None, top_k=None)
gen = out[0, ids.shape[1]:]
print("CPU-bf16 token ids:", gen.tolist(), flush=True)
print("CPU-bf16 OUTPUT:", repr(tok.decode(gen)), flush=True)
