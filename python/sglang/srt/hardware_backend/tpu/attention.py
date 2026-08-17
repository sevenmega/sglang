"""TPU attention backend for SGLang.

Phase 1: Delegates entirely to TorchNativeAttnBackend (SDPA on CPU tensors).
Phase 2: Will use PPL-compiled attention kernels on TPU HBM for decode/extend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class TpuAttnBackend(TorchNativeAttnBackend):
    """TPU attention backend.

    Currently inherits TorchNativeAttnBackend unchanged. This subclass exists
    as the extension point for PPL-accelerated attention kernels in Phase 2.
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
