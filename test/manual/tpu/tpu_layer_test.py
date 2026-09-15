"""TPU single-layer replay: run one Qwen3 decoder layer on the SG2260E device
and compare its output against the CPU golden reference (cpu_layer_dump.py).

Loads the same Qwen3-0.6B in bf16, extracts ``model.model.layers[N]``, moves it
onto the ``tpu`` device, and runs it on the *exact* inputs the CPU reference fed
its layer N (hidden_states, attention_mask, position_ids, position_embeddings).
The only device concession is attention: the device can't do 4D bmm or SDPA, so
we register a TPU-safe per-head 2D-matmul attention (mirroring the real
``TpuAttnBackend`` math) and point the layer's config at it. Everything else
(RMSNorm, q/k/v/o projections, q/k per-head norm, RoPE, MLP) runs as stock HF
ops on the device.

Env: LAYER (default 0), REF_OUT (default /tmp/layer_ref.pt), device via
SGLANG_TPU_DEVICE_ID (default 4). Run under /tmp/tpu_env.sh.
"""

import os

import torch
import torch_tpu  # noqa: F401  (registers the "tpu" device)
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

MODEL = os.environ.get("MODEL_PATH", "/workspace/Qwen3-0.6B")
LAYER = int(os.environ.get("LAYER", "0"))
REF = os.environ.get("REF_OUT", "/tmp/layer_ref.pt")
DEV_ID = int(os.environ.get("SGLANG_TPU_DEVICE_ID", "4"))
DEV = f"tpu:{DEV_ID}"

torch.tpu.set_device(DEV_ID)


# ---------------------------------------------------------------------------
# TPU-safe attention: per-head loop of contiguous 2D bf16 matmuls, additive
# mask. Mirrors sglang's TpuAttnBackend._attend_per_req. HF's Qwen3Attention
# hands us query [B,Hq,Sq,D], key/value [B,Hkv,Skv,D] and an additive mask, and
# expects the output shaped [B,Sq,Hq,D] (it reshapes then applies o_proj).
# ---------------------------------------------------------------------------
def tpu_perhead_attention(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    B, Hq, Sq, D = query.shape
    Hkv = key.shape[1]
    group = Hq // Hkv
    out = torch.empty_like(query)  # [B, Hq, Sq, D]

    for b in range(B):
        for h in range(Hq):
            kv_h = h // group
            q_i = query[b, h].contiguous()  # [Sq, D]
            k_i = key[b, kv_h].contiguous()  # [Skv, D]
            v_i = value[b, kv_h].contiguous()  # [Skv, D]

            scores = q_i @ k_i.t().contiguous()  # [Sq, Skv] bf16 2D matmul
            scores = scores.float() * scaling
            if attention_mask is not None:
                m = attention_mask[b, 0].float()  # [Sq, Skv]
                scores = scores + m[:, : k_i.shape[0]]
            probs = torch.softmax(scores, dim=-1).to(v_i.dtype)
            out[b, h] = probs.contiguous() @ v_i  # [Sq, D]

    # eager returns [B, Sq, Hq, D]; caller reshapes to [B, Sq, Hq*D].
    return out.transpose(1, 2).contiguous(), None


ALL_ATTENTION_FUNCTIONS.register("tpu_perhead", tpu_perhead_attention)


# ---------------------------------------------------------------------------
# Build the layer on CPU (weights), then move it to the device.
# ---------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager"
)
model.eval()
layer = model.model.layers[LAYER]
layer.self_attn.config._attn_implementation = "tpu_perhead"
layer = layer.to(DEV)

ref = torch.load(REF, weights_only=False)
assert ref["layer"] == LAYER, f"reference is for layer {ref['layer']}, not {LAYER}"


def _to_dev(t):
    return t.to(DEV) if isinstance(t, torch.Tensor) else t


hidden_states = _to_dev(ref["hidden_states"])
attention_mask = _to_dev(ref["attention_mask"])
position_ids = _to_dev(ref["position_ids"])
pe = ref["position_embeddings"]
position_embeddings = (_to_dev(pe[0]), _to_dev(pe[1])) if pe is not None else None

with torch.no_grad():
    out = layer(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        use_cache=False,
    )
tpu_out = (out[0] if isinstance(out, tuple) else out).float().cpu()
ref_out = ref["output"].float()

# ---------------------------------------------------------------------------
# Compare (bf16 tolerance).
# ---------------------------------------------------------------------------
diff = (tpu_out - ref_out).abs()
# Norm-based relative error ||tpu - ref|| / ||ref||: robust to the many
# near-zero elements that make a per-element mean relative error meaningless.
rel = (diff.norm() / ref_out.norm().clamp_min(1e-6)).item()
cos = torch.nn.functional.cosine_similarity(
    tpu_out.flatten(), ref_out.flatten(), dim=0
).item()

print(f"=== TPU layer {LAYER} replay vs CPU reference ===")
print(f"  shape          : tpu {tuple(tpu_out.shape)}  ref {tuple(ref_out.shape)}")
print(f"  max abs err    : {diff.max().item():.6f}")
print(f"  mean abs err   : {diff.mean().item():.6f}")
print(f"  rel err (norm) : {rel:.4%}")
print(f"  ref abs mean   : {ref_out.abs().mean().item():.6f}")
print(f"  cosine sim     : {cos:.6f}")
# Scale-invariant bf16 tolerance: the residual stream grows through depth and
# has large-magnitude outlier dims, so absolute error is not comparable across
# layers. Norm-relative error + cosine are the meaningful checks.
PASS = cos > 0.999 and rel < 0.02
print(f"\n  {'PASS' if PASS else 'FAIL'} (cosine>0.999 and rel_err<2%)")
