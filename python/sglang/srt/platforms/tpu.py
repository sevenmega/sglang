"""TPU (Sophgo SG2260E) device operations for the SRT platform layer."""

from __future__ import annotations

import ctypes
import logging
import os
from typing import Optional

import torch

from sglang.srt.platforms.device_mixin import (
    DeviceCapability,
    DeviceMixin,
    PlatformEnum,
)
from sglang.srt.platforms.interface import SRTPlatform

logger = logging.getLogger(__name__)

# Sophgo TPU HBM size (32 GB for SG2260E, configurable via env)
_TPU_HBM_SIZE = int(os.environ.get("SGLANG_TPU_HBM_SIZE", str(32 * 1024**3)))
_TPU_DEVICE_ID = int(os.environ.get("SGLANG_TPU_DEVICE_ID", "2"))


def _get_tpu_runtime_lib():
    """Attempt to load the TPU runtime library for memory queries."""
    ppl_runtime_path = os.environ.get("PPL_RUNTIME_PATH", "")
    lib_path = os.path.join(ppl_runtime_path, "libtpurt.so")
    if os.path.exists(lib_path):
        try:
            return ctypes.CDLL(lib_path)
        except OSError:
            pass
    return None


class TpuDeviceMixin(DeviceMixin):
    """Sophgo TPU implementation of the shared device operations.

    The Sophgo SG2260E TPU is accessed via the PPL toolchain and ctypes
    bindings to libtpurt.so / libtpudnn.so. Tensors live on the host (CPU)
    and are transferred to/from TPU HBM for kernel execution.

    Since torch does not have a native "tpu" device backend for Sophgo chips,
    we use "cpu" as the torch device type and manage TPU memory explicitly
    via the PPL runtime.
    """

    _enum: PlatformEnum = PlatformEnum.TPU
    device_name: str = "tpu"
    device_type: str = "cpu"  # torch tensors live on CPU; TPU memory managed via PPL

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return _TPU_HBM_SIZE

    def get_current_memory_usage(
        self, device: Optional[torch.device] = None
    ) -> float:
        # No torch-level tracking for TPU; return 0 for now.
        # Future: query PPL runtime for allocated TPU memory.
        return 0.0

    def get_device(self, local_rank: int) -> torch.device:
        # Sophgo TPU tensors are managed on CPU; actual TPU execution
        # happens via explicit DMA in the PPL runtime.
        return torch.device("cpu")

    def set_device(self, device: torch.device) -> None:
        pass  # No-op: TPU device selection is via PPL's devid parameter

    def get_device_name(self, device_id: int = 0) -> str:
        return f"sophgo-sg2260e (devid={_TPU_DEVICE_ID})"

    def get_device_uuid(self, device_id: int = 0) -> str:
        return f"sophgo-tpu-{device_id}"

    def get_device_capability(self, device_id: int = 0) -> Optional[DeviceCapability]:
        return None

    def empty_cache(self) -> None:
        pass  # TPU memory managed by PPL runtime

    def synchronize(self) -> None:
        pass  # PPL kernel launches are synchronous

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        # Conservative: report full HBM as available (PPL manages internally)
        return (_TPU_HBM_SIZE, _TPU_HBM_SIZE)

    def is_pin_memory_available(self, device=None) -> bool:
        return False

    def get_torch_distributed_backend_str(self) -> str:
        return "gloo"


class TpuSRTPlatform(TpuDeviceMixin, SRTPlatform):
    """Sophgo TPU SRT platform.

    Phase 1 implementation:
    - Uses torch_native attention backend (SDPA on CPU tensors)
    - No CUDA graph support
    - No FP8 support
    - Standard MHA KV pool (CPU-backed)
    - Gloo for distributed communication

    Future phases will add:
    - PPL-accelerated attention kernels
    - TPU HBM-backed KV cache pools
    - Tensor parallelism via TPU interconnect
    """

    def get_default_attention_backend(self) -> str:
        return "torch_native"

    def get_graph_runner_cls(self) -> type:
        raise NotImplementedError("TPU does not support graph capture/replay")

    def get_mha_kv_pool_cls(self) -> type:
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        return MHATokenToKVPool

    def get_mla_kv_pool_cls(self) -> type:
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        return MLATokenToKVPool

    def get_paged_allocator_cls(self) -> type:
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

        return PagedTokenToKVPoolAllocator

    def supports_fp8(self) -> bool:
        return False

    def support_cuda_graph(self) -> bool:
        return False

    def support_piecewise_cuda_graph(self) -> bool:
        return False

    def apply_server_args_defaults(self, server_args) -> None:
        """Apply TPU-specific defaults."""
        # Disable chunked prefill by default (no async overlap on TPU)
        if server_args.chunked_prefill_size is None:
            server_args.chunked_prefill_size = -1
        # Disable CUDA graph (not supported)
        server_args.disable_cuda_graph = True

    def init_backend(self) -> None:
        """Initialize PPL runtime if available."""
        ppl_root = os.environ.get("PPL_PROJECT_ROOT")
        if ppl_root:
            logger.info("TPU backend: PPL_PROJECT_ROOT=%s", ppl_root)
        else:
            logger.warning(
                "TPU backend: PPL_PROJECT_ROOT not set. "
                "Hardware execution will not be available."
            )

    def get_dispatch_key_name(self) -> str:
        return "native"


def activate():
    """Entry point for platform plugin discovery.

    Returns the fully-qualified class name if TPU hardware is detected,
    or None otherwise.
    """
    # Check if user explicitly requests TPU via environment
    if os.environ.get("SGLANG_USE_TPU", "0") == "1":
        return "sglang.srt.platforms.tpu.TpuSRTPlatform"

    # Auto-detect: check if PPL runtime is available
    ppl_root = os.environ.get("PPL_PROJECT_ROOT")
    if ppl_root and os.path.isdir(ppl_root):
        return "sglang.srt.platforms.tpu.TpuSRTPlatform"

    return None
