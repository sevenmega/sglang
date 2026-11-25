import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mp = "/workspace/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(mp)
model = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.bfloat16).eval()

base = tok("The capital of France is").input_ids
prefix = base + [12095, 13, 576, 6722, 315]  # + " Paris. The capital of"
ids = torch.tensor([prefix])
with torch.no_grad():
    logits = model(ids).logits[0, -1].float()
lp = torch.log_softmax(logits, dim=-1)
for tid, name in [(15344, "Italy"), (9625, "France")]:
    print(f"CPU id={tid} ({name}) lp={lp[tid].item():.4f}", flush=True)
print("CPU argmax:", lp.argmax().item(), flush=True)
