#!/usr/bin/env bash
# Install SGLang for the Sophgo SG2260E TPU (torch_tpu backend), CUDA-free.
#
# Prerequisites (installed out-of-band, NOT via pyproject):
#   - torch (CPU build matching torch_tpu, e.g. 2.10.0+cpu)
#   - torch_tpu  (https://github.com/sophgo/torch-tpu, editable at /workspace/tpu-train)
#
# torch_tpu registers a real "tpu" PyTorch device (torch.tpu, mirroring torch.cuda).
set -euo pipefail

SGLANG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SGLANG_DIR/python"

# Swap in the TPU pyproject (light deps, no CUDA, no Rust extensions).
cp pyproject_tpu.toml pyproject.toml

# Build without any Rust extensions or sgl-kernel.
SGLANG_BUILD_RUST_EXTS=none pip install -e . --no-build-isolation

echo
echo "Installed SGLang (TPU). Verify with:"
echo "  SGLANG_USE_TPU=1 python -c 'import torch_tpu, sglang; print(sglang.__version__)'"
