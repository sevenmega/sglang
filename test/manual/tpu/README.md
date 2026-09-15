# Manual TPU (Sophgo SG2260E) verification scripts

Manual, hardware-dependent scripts used to bring up and verify SGLang on the
Sophgo SG2260E TPU (the real `"tpu"` torch device provided by `torch_tpu`).
They are **not** part of the automated pytest suite — they require a physical
TPU and a local model checkout, and are run by hand during bring-up / debugging.

The guiding idea: **HF `transformers` on CPU is the golden reference.** SGLang's
CPU engine can't run on this host (it hard-depends on a compiled `sgl_kernel`),
so every TPU result is validated against a stock HF `transformers` run in
`bfloat16`, which shares only the model weights — not any serving code — with
the TPU path.

## Dependencies

Common to all scripts:

- **Model weights** at `/workspace/Qwen3-0.6B` (override with `MODEL_PATH` where
  the script supports it). Qwen3-0.6B is the bring-up model.
- **`torch` + `transformers`** (already required by SGLang).

CPU-reference scripts (`cpu_*.py`) need only the above — they run pure HF
`transformers` on CPU and touch no device.

TPU scripts (`tpu_*.py`) additionally need:

- **`torch_tpu`** installed (registers the `"tpu"` device and `torch.tpu`).
- **SGLang installed editable** for the TPU platform
  (`cp python/pyproject_tpu.toml python/pyproject.toml && pip install -e python`).
- **The TPU runtime + PPL environment** on `LD_LIBRARY_PATH`, and the target
  device selected. All of this is captured in [`tpu_env.sh`](./tpu_env.sh) —
  source it before running any `tpu_*.py` script:

  ```bash
  source test/manual/tpu/tpu_env.sh
  ```

  Adjust `PPL_PROJECT_ROOT` and `SGLANG_TPU_DEVICE_ID` (which device of the 8 to
  use; default 4) in that file for your host. It also sets `SGLANG_USE_TPU=1` and
  puts `python/` on `PYTHONPATH`.

## Scripts

| Script | Device | Purpose |
|---|---|---|
| `cpu_ref.py` | CPU | Golden reference: greedy 16-token generation for `"The capital of France is"`. Prints token ids + text. |
| `tpu_lp16.py` | TPU | Same greedy generation through an `sgl.Engine` on the TPU. Compare its ids against `cpu_ref.py`. |
| `cpu_tie.py` | CPU | Probes the last-token logprobs of `Italy` vs `France` at the known bf16-tie decode step (the point where TPU and CPU greedy paths legitimately diverge). |
| `tpu_tie.py` | TPU | Same logprob probe on the TPU (via `token_ids_logprob`, since `top_logprobs_num`/topk is unimplemented on-device). Confirms the divergence is a genuine bf16 tie, not a bug. |
| `cpu_layer_dump.py` | CPU | Dumps one Qwen3 decoder layer's inputs + output (via forward hooks) to `/tmp/layer_ref.pt` as the per-layer golden reference. |
| `tpu_layer_test.py` | TPU | Replays that **same** decoder layer on the TPU from the dumped inputs and compares the output (bf16 tolerance). Per-layer correctness probe. |

## Usage

### 1. End-to-end generation (TPU vs CPU)

```bash
# CPU golden reference (no TPU needed)
python test/manual/tpu/cpu_ref.py
#   CPU-bf16 token ids: [12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, ...]

# TPU end-to-end
source test/manual/tpu/tpu_env.sh
python test/manual/tpu/tpu_lp16.py
```

The first ~5 tokens match; the paths then diverge at a genuine bf16 logit tie
(`Italy` vs `France`), which the tie probe below confirms is not a bug.

### 2. bf16-tie probe

```bash
python test/manual/tpu/cpu_tie.py                    # CPU: both ≈ -1.7736 (tie)
source test/manual/tpu/tpu_env.sh
python test/manual/tpu/tpu_tie.py                    # TPU: within ~0.07 nat
```

### 3. Single-layer replay (per-layer correctness)

Dump on CPU, replay on TPU, for a chosen decoder layer index:

```bash
# 1) dump layer N's I/O as the golden reference (CPU, no TPU)
LAYER=0 python test/manual/tpu/cpu_layer_dump.py     # -> /tmp/layer_ref.pt

# 2) replay the same layer on the TPU and compare
source test/manual/tpu/tpu_env.sh
LAYER=0 python test/manual/tpu/tpu_layer_test.py
```

`LAYER` (default 0), `PROMPT`, and `REF_OUT` (default `/tmp/layer_ref.pt`) are
env-configurable and must match between the two steps. The replay runs the layer
as stock HF ops on the device; the only device concession is a TPU-safe per-head
2D-matmul attention (the device can't do 4D bmm / SDPA), mirroring the real
`TpuAttnBackend` math.

Pass criterion is scale-invariant (bf16): **cosine > 0.999** and
**norm-relative error < 2%**. Absolute error is not comparable across layers —
the residual stream grows through depth and has large-magnitude outlier
dimensions, so a fixed absolute threshold spuriously fails deep layers.

Example (layers 0 / 14 / 27, all PASS): rel err 1.47% / 0.01% / 1.62%,
cosine 0.99990 / 1.00000 / 0.99994.

## Notes / device constraints

These scripts encode empirically-verified SG2260E constraints (see
`python/sglang/srt/platforms/tpu.py` and `hardware_backend/tpu/attention.py`):

- matmul is correct only in **bf16**; operands must be `.contiguous()`;
- **batched matmul (3D/4D bmm) is broken** → attention loops per head over 2D matmuls;
- `F.scaled_dot_product_attention` is unimplemented;
- `topk` is unimplemented → the tie probe uses `token_ids_logprob` gather;
- on-device integer arithmetic is unreliable → decode positions are computed on host.
