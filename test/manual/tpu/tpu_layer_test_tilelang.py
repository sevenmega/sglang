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
SGLANG_TPU_DEVICE_ID (default 4). Run under tpu_env.sh. Set
SGLANG_DEBUG_TPU_TRACE=1 to print a ``[TPU-OP]`` line per aten op in this
layer's forward (op type + input/output shapes) for debugging/optimization.
"""

import contextlib
import os
import sys

import torch
import torch_tpu  # noqa: F401  (registers the "tpu" device)
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

# tilelang lives in a dev checkout (loads its native PPL libs from there).
if "/workspace/tilelang" not in sys.path:
    sys.path.insert(0, "/workspace/tilelang")
import tilelang.tpu as tpu  # noqa: E402

MODEL = os.environ.get("MODEL_PATH", "/workspace/Qwen3-0.6B")
LAYER = int(os.environ.get("LAYER", "0"))
REF = os.environ.get("REF_OUT", "/tmp/layer_ref.pt")
DEV_ID = int(os.environ.get("SGLANG_TPU_DEVICE_ID", "4"))
DEV = f"tpu:{DEV_ID}"

torch.tpu.set_device(DEV_ID)

# fp16-safe stand-in for -inf in the additive mask (-1e9 overflows fp16, which
# the flash kernel computes in internally). Any masked position gets this.
_MASK_NEG = -30000.0

# Module-cached flash-attention kernels keyed by shape. Compiling invokes
# ppl-compile (seconds) so we build once per (B,Sq,Skv,D,Hq,Hkv).
_FLASH_KERNELS: dict[tuple, object] = {}


def _causal_mask(b, sq, skv, device):
    """Additive [b, sq, skv] fp32 causal mask, clamped fp16-safe."""
    i = torch.arange(sq, device=device).unsqueeze(1)
    j = torch.arange(skv, device=device).unsqueeze(0)
    disallowed = j > (i + (skv - sq))
    mask = torch.zeros(b, sq, skv, dtype=torch.float32, device=device)
    mask.masked_fill_(disallowed.unsqueeze(0), _MASK_NEG)
    return mask


def _dtype_name(dt):
    # str(torch.bfloat16) is "torch.bfloat16"; T.dtype wants "bfloat16".
    return str(dt).replace("torch.", "")


def _get_flash_kernel(d, scaling, dtype):
    # Shape-generic kernel: (d, scaling, dtype) are the compile-time params, so
    # key on those; B/Hq/Sq/Hkv/Skv resolve per launch.
    key = (d, float(scaling), _dtype_name(dtype))
    kern = _FLASH_KERNELS.get(key)
    if kern is None:
        from tilelang.tpu.kernels.attention import flash_attention_gqa

        kern = flash_attention_gqa(
            d=d, sm_scale=float(scaling), dtype=_dtype_name(dtype)
        )
        _FLASH_KERNELS[key] = kern
    return kern


# ---------------------------------------------------------------------------
# TPU flash attention: online-softmax GQA kernel implemented as a real tilelang
# DSL kernel (tilelang.tpu.kernels.attention.flash_attention_gqa) compiled
# through the generic TIR->PPL translator. The kernel replaces the per-head
# 2D-matmul loop. HF hands us query [B,Hq,Sq,D], key/value [B,Hkv,Skv,D] and an
# additive mask [B,1,Sq,Skv]; it expects output [B,Sq,Hq,D]. Our kernel is
# head-major: q [B,Hq,Sq,D] fp16, k/v [B,Hkv,Skv,D] fp16, mask [B,Sq,Skv] fp32
# additive, output [B,Hq,Sq,D] fp16.
#
# The kernel has fixed on-chip tiles (8 query rows, 128 keys) and no
# partial-tile clamping, so Sq/Skv are padded up and the output is sliced back.
# ---------------------------------------------------------------------------
def tpu_perhead_attention(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    B, Hq, Sq, D = query.shape
    Hkv, Skv = key.shape[1], key.shape[2]
    orig_dtype, orig_device = query.dtype, query.device

    block_m, block_n = 8, 128
    q_pad = -(-Sq // block_m) * block_m
    kv_pad = -(-Skv // block_n) * block_n
    kv_pad = max(kv_pad, q_pad)

    # Device tensors throughout: the kernel launches with device pointers, so
    # the padding and the mask are built where the data already lives. Nothing
    # is marshalled through CPU memory, and the dtype stays whatever HF gave us
    # (the kernel compiles for it directly).
    q = torch.zeros(B, Hq, q_pad, D, dtype=orig_dtype, device=orig_device)
    q[:, :, :Sq] = query.detach()
    k = torch.zeros(B, Hkv, kv_pad, D, dtype=orig_dtype, device=orig_device)
    k[:, :, :Skv] = key.detach()
    v = torch.zeros(B, Hkv, kv_pad, D, dtype=orig_dtype, device=orig_device)
    v[:, :, :Skv] = value.detach()

    # Additive mask [B,1,Sq,Skv] -> padded [B,q_pad,kv_pad] fp32, fp16-safe.
    if attention_mask is not None:
        m = attention_mask[:, :1, :, :Skv].float().to(orig_device)  # [B,1,Sq,Skv]
        m = m.squeeze(1).clamp_min(_MASK_NEG)  # [B,Sq,Skv]
    else:  # causal fallback
        m = _causal_mask(B, Sq, Skv, orig_device)  # [B,Sq,Skv]

    if q_pad != Sq or kv_pad != Skv:
        # Mask row r <-> query row r, column j <-> key j (the kernel bakes in no
        # causal constraint). Padded key columns are -30000 so no real query
        # attends a padded key; padded query rows are 0 because the kernel
        # softmaxes over a whole query tile.
        padded = torch.full(
            (B, q_pad, kv_pad), _MASK_NEG, dtype=torch.float32, device=orig_device
        )
        padded[:, :Sq, :Skv] = m
        padded[:, Sq:, :] = 0.0
        m = padded

    kernel = _get_flash_kernel(D, scaling, orig_dtype)
    out = kernel(q, k, v, m)  # [B,Hq,q_pad,D], on device
    out = out[:, :, :Sq]  # [B,Hq,Sq,D]

    # HF's eager attention returns [B, Sq, Hq, D]; the caller reshapes to
    # [B, Sq, Hq*D].
    out = out.transpose(1, 2).contiguous()  # [B,Sq,Hq,D]
    return out.to(orig_dtype).to(orig_device), None


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

# Optional op-level trace of *this single layer's* forward
# (SGLANG_DEBUG_TPU_TRACE=1). This script drives the layer directly and never
# calls the SGLang TPU platform's init_backend(), which is where serving runs
# install the tracer -- so we install it here ourselves, and scope it to just
# the forward() so the trace is the layer's compute ops (no weight-copy noise).
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.trace import TpuOpTracer

trace_cm = TpuOpTracer() if envs.SGLANG_DEBUG_TPU_TRACE.get() else contextlib.nullcontext()

with torch.no_grad(), trace_cm:
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
