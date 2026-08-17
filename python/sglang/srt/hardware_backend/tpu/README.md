# SGLang TPU Backend (Sophgo SG2260E)

This backend enables SGLang to serve LLM inference on Sophgo SG2260E TPU hardware via the PPL toolchain.

## Architecture

The TPU backend integrates at two levels:

1. **Platform layer** (`sglang/srt/platforms/tpu.py`): Implements `TpuSRTPlatform` — device operations, memory management, and SRT factory methods. Torch tensors live on CPU; TPU execution happens via explicit DMA through the PPL runtime.

2. **Attention backend** (`sglang/srt/hardware_backend/tpu/attention.py`): Phase 1 delegates to `TorchNativeAttnBackend` (PyTorch SDPA). Phase 2 will integrate PPL-compiled attention kernels for decode/extend on TPU HBM.

## Requirements

- Python 3.10+
- PyTorch (CPU build is sufficient for Phase 1)
- PPL toolchain (`PPL_PROJECT_ROOT` pointing to a valid PPL installation)
- Sophgo SG2260E hardware (for actual TPU execution)

## Setup

```bash
# Set PPL environment
export PPL_PROJECT_ROOT=/path/to/ppl
export PPL_RUNTIME_PATH=$PPL_PROJECT_ROOT/runtime/lib

# Optional: explicit TPU device ID (default: 2)
export SGLANG_TPU_DEVICE_ID=2

# Optional: override reported HBM size (default: 32GB)
export SGLANG_TPU_HBM_SIZE=34359738368

# Install SGLang
cd /path/to/sglang/python
pip install -e .
```

## Usage

### Automatic detection

The TPU platform activates automatically when `PPL_PROJECT_ROOT` is set and points to a valid directory:

```bash
export PPL_PROJECT_ROOT=/opt/ppl
python -m sglang.launch_server --model-path /path/to/model
```

### Explicit activation

Force TPU mode regardless of PPL availability:

```bash
export SGLANG_USE_TPU=1
python -m sglang.launch_server --model-path /path/to/model
```

### Plugin-style selection

Use `SGLANG_PLATFORM` for explicit entry-point-based selection:

```bash
export SGLANG_PLATFORM=tpu
python -m sglang.launch_server --model-path /path/to/model
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_USE_TPU` | `0` | Set to `1` to force TPU platform activation |
| `SGLANG_PLATFORM` | (unset) | Set to `tpu` for entry-point plugin selection |
| `PPL_PROJECT_ROOT` | (unset) | Path to PPL toolchain; triggers auto-detection |
| `PPL_RUNTIME_PATH` | (unset) | Path to PPL runtime libraries (`libtpurt.so`) |
| `SGLANG_TPU_DEVICE_ID` | `2` | TPU device ID for PPL runtime calls |
| `SGLANG_TPU_HBM_SIZE` | `34359738368` | Reported HBM capacity in bytes (32 GB) |

## Phase 1 Limitations

- Tensors on CPU only (no direct TPU tensor allocation via PyTorch)
- Attention via PyTorch SDPA (not hardware-accelerated)
- No CUDA graph support
- No FP8 quantization
- Distributed backend: gloo (no NCCL)
- No chunked prefill overlap

## Roadmap

- **Phase 2**: PPL-accelerated attention kernels operating directly on TPU HBM
- **Phase 3**: TPU HBM-backed KV cache pools with DMA transfers
- **Phase 4**: Tensor parallelism via TPU interconnect
