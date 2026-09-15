"""TPU (Sophgo SG2260E) device operations for the SRT platform layer.

Unlike the earlier scaffold that treated TPU as a CPU-backed device, this
platform drives the *real* ``"tpu"`` torch device provided by ``torch_tpu``
(https://github.com/sophgo/torch-tpu). ``torch_tpu`` registers a privateuseone
backend renamed to ``"tpu"`` and mirrors ``torch.cuda`` as ``torch.tpu``, so
device management, memory queries and tensor placement all go through
``torch.tpu`` / ``torch.get_device_module("tpu")``.

Device numerical constraints (empirically verified on hardware) shape the
serving defaults and the dedicated ``"tpu"`` attention backend:

- matmul is correct only in bf16 (fp32/fp16 return garbage) → force bfloat16;
- batched matmul (3D/4D bmm) is broken → the attention backend loops per head
  over contiguous 2D matmuls;
- ``F.scaled_dot_product_attention`` is unimplemented → custom attention math.
"""

from __future__ import annotations

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


def _import_torch_tpu() -> None:
    """Import torch_tpu, which registers the ``"tpu"`` device and ``torch.tpu``.

    Idempotent: torch_tpu guards its own re-import, and ``torch.tpu`` simply
    stays present once registered.
    """
    import torch_tpu  # noqa: F401


_arange_shim_installed = False


def _install_tpu_arange_shim() -> None:
    """Make ``torch.arange(..., device="tpu")`` tolerate int64.

    The device rejects ``torch.arange`` with an int64 dtype ("arange only
    support int32 & float32 now"), yet SGLang creates index tensors (KV-cache
    free lists, ``req_to_token``, page offsets, ...) with the int64 default all
    over the hot path. A host→device copy of an int64 tensor is accepted (it is
    materialized as int32, which is what indexing uses anyway), so for TPU
    targets we build the range on CPU and copy it over. Non-TPU devices are
    untouched. Idempotent.
    """
    global _arange_shim_installed
    if _arange_shim_installed:
        return

    _orig_arange = torch.arange

    def _arange(*args, **kwargs):
        device = kwargs.get("device")
        if device is not None and "tpu" in str(device):
            # tpu scalar tensors as arange bounds are unsupported; coerce any
            # 0-dim tensor start/stop/step to Python numbers first.
            args = tuple(
                a.item() if isinstance(a, torch.Tensor) else a for a in args
            )
            dtype = kwargs.get("dtype")
            if dtype is None or dtype == torch.int64:
                host_kwargs = dict(kwargs)
                host_kwargs.pop("device", None)
                return _orig_arange(*args, **host_kwargs).to(device)
            return _orig_arange(*args, **kwargs)
        return _orig_arange(*args, **kwargs)

    torch.arange = _arange
    _arange_shim_installed = True


_blocking_copy_shim_installed = False


def _install_tpu_blocking_copy_shim() -> None:
    """Force ``non_blocking=True`` host<->device copies to be synchronous.

    ``torch_tpu``'s async H2D copy reads the *source* CPU tensor after the
    Python statement returns. SGLang's hot path is full of
    ``torch.tensor(list).to(device, non_blocking=True)`` — the temporary CPU
    tensor is freed before the async copy lands, so the device tensor ends up
    holding garbage (verified: repeated ``[0, 6]`` copies yield ``[505560067,
    0]``). ``non_blocking=False`` is always semantically correct (it only gives
    up copy/compute overlap), so we strip the flag for every ``.to`` / ``.copy_``
    once torch_tpu is active. Idempotent.
    """
    global _blocking_copy_shim_installed
    if _blocking_copy_shim_installed:
        return

    _orig_to = torch.Tensor.to
    _orig_copy_ = torch.Tensor.copy_

    def _to(self, *args, **kwargs):
        if kwargs.get("non_blocking"):
            kwargs = dict(kwargs)
            kwargs["non_blocking"] = False
        return _orig_to(self, *args, **kwargs)

    def _copy_(self, src, non_blocking=False):
        return _orig_copy_(self, src, non_blocking=False)

    torch.Tensor.to = _to
    torch.Tensor.copy_ = _copy_
    _blocking_copy_shim_installed = True


class TpuDeviceMixin(DeviceMixin):
    """Sophgo TPU implementation of the shared device operations.

    All operations are delegated to ``torch.tpu`` (== ``torch.get_device_module
    ("tpu")``), the CUDA-mirroring module that ``torch_tpu`` installs.
    """

    _enum: PlatformEnum = PlatformEnum.TPU
    device_name: str = "tpu"
    device_type: str = "tpu"

    @staticmethod
    def _module():
        # torch.tpu exists once torch_tpu has been imported (init_backend /
        # activate guarantee this before any device op runs).
        return torch.get_device_module("tpu")

    def get_device_total_memory(self, device_id: int = 0) -> int:
        _, total = self._module().mem_get_info(device_id)
        return total

    def get_current_memory_usage(
        self, device: Optional[torch.device] = None
    ) -> float:
        mod = self._module()
        device_id = device.index if isinstance(device, torch.device) else device
        mod.reset_peak_memory_stats(device_id)
        return mod.max_memory_allocated(device_id)

    def get_device(self, device_id: Optional[int] = None) -> str:
        if device_id is None:
            return "tpu"
        return f"tpu:{device_id}"

    def set_device(self, device: torch.device) -> None:
        self._module().set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        return self._module().get_device_name(device_id)

    def get_device_uuid(self, device_id: int = 0) -> str:
        return f"sophgo-tpu-{device_id}"

    def get_device_capability(self, device_id: int = 0) -> Optional[DeviceCapability]:
        return None

    def empty_cache(self) -> None:
        self._module().empty_cache()

    def synchronize(self) -> None:
        self._module().synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        free, total = self._module().mem_get_info(device_id)
        return (free, total)

    def is_pin_memory_available(self, device=None) -> bool:
        return False


class TpuSRTPlatform(TpuDeviceMixin, SRTPlatform):
    """Sophgo TPU SRT platform (real ``"tpu"`` torch device via torch_tpu).

    - bf16-only compute; a dedicated per-head 2D-matmul ``"tpu"`` attention
      backend (no SDPA, no bmm);
    - no CUDA graph / piecewise graph, no fp8;
    - standard paged MHA KV pool allocated on the ``"tpu"`` device;
    - gloo for host-side collectives (single-device TP=1 is the default).
    """

    def get_default_attention_backend(self) -> str:
        return "tpu"

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
        """Force the TPU-safe serving configuration.

        The device only produces correct results in bf16 with the dedicated
        ``"tpu"`` attention backend and pytorch sampling; graph capture and the
        async overlap scheduler are not supported.
        """
        # bf16 is the only numerically-correct matmul dtype on this device.
        if server_args.dtype in (None, "auto"):
            server_args.dtype = "bfloat16"

        # Route all attention through the TPU per-head 2D-matmul backend.
        if server_args.attention_backend is None:
            server_args.attention_backend = "tpu"
        if server_args.prefill_attention_backend is None:
            server_args.prefill_attention_backend = "tpu"
        if server_args.decode_attention_backend is None:
            server_args.decode_attention_backend = "tpu"

        # Pure-pytorch sampling (no sgl-kernel / flashinfer sampling on TPU).
        server_args.sampling_backend = "pytorch"

        # No graph capture, no async overlap, no custom all-reduce on TPU. The
        # cuda_graph_config was already resolved (before this hook runs) from the
        # then-default disable_cuda_graph=False, so disable both phases directly
        # on the config as well — mirroring the XPU/NPU handlers.
        from sglang.srt.model_executor.cuda_graph_config import Backend, Phase

        server_args.disable_cuda_graph = True
        server_args.cuda_graph_config.prefill.backend = Backend.DISABLED
        server_args.cuda_graph_config.decode.backend = Backend.DISABLED
        server_args.disable_overlap_schedule = True
        server_args.enable_custom_all_reduce = False

        # The eager runner otherwise copies each live batch into fixed static
        # input buffers via ``torch._foreach_copy_`` (a leftover from graph
        # capture, which TPU never does). That copy is pure overhead here and
        # trips the device's stricter ``_foreach_copy_`` (which rejects a
        # dst/src byte-size mismatch that arises from int64 index tensors being
        # materialized as int32). Feed live tensors straight through instead.
        # Respect an explicit user setting; only seed the default. The
        # scheduler subprocess inherits this via os.environ.
        from sglang.srt.environ import envs

        if not envs.SGLANG_EAGER_INPUT_NO_COPY.is_set():
            envs.SGLANG_EAGER_INPUT_NO_COPY.set(True)

    def init_backend(self) -> None:
        """Import torch_tpu (registers ``torch.tpu``) and select the device."""
        _import_torch_tpu()
        _install_tpu_arange_shim()
        _install_tpu_blocking_copy_shim()
        device_id = int(os.environ.get("SGLANG_TPU_DEVICE_ID", "0"))
        torch.get_device_module("tpu").set_device(device_id)

        # Optional op-level trace (SGLANG_DEBUG_TPU_TRACE=1): log every aten op
        # dispatched to the device with its input/output shapes.
        from sglang.srt.hardware_backend.tpu.trace import maybe_enable_tpu_op_trace

        maybe_enable_tpu_op_trace()
        logger.info(
            "TPU backend initialized: %s (device %d of %d)",
            self.get_device_name(device_id),
            device_id,
            torch.get_device_module("tpu").device_count(),
        )

    def get_dispatch_key_name(self) -> str:
        return "native"


def activate():
    """Entry point for platform plugin discovery.

    Returns the fully-qualified platform class name when TPU is requested
    (``SGLANG_USE_TPU=1`` or a PPL project root is present) and torch_tpu can
    be imported; otherwise ``None``.
    """
    requested = os.environ.get("SGLANG_USE_TPU", "0") == "1" or bool(
        os.environ.get("PPL_PROJECT_ROOT")
    )
    if not requested:
        return None

    try:
        _import_torch_tpu()
    except ImportError:
        logger.warning(
            "SGLANG_USE_TPU requested but 'torch_tpu' could not be imported; "
            "TPU platform not activated."
        )
        return None

    if not torch.get_device_module("tpu").is_available():
        logger.warning("torch_tpu imported but no TPU device is available.")
        return None

    return "sglang.srt.platforms.tpu.TpuSRTPlatform"
