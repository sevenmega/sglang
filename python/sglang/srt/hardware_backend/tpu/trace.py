"""Op-level tracer for the Sophgo SG2260E TPU device.

Enabled by ``SGLANG_DEBUG_TPU_TRACE=1``: installs a process-wide
:class:`TorchDispatchMode` that prints one ``[TPU-OP]`` line per aten op whose
inputs or outputs live on the ``"tpu"`` device, with the op type and the shape
(and dtype) of every tensor argument and result.

This is the single, comprehensive answer to "print each call into the TPU":
``__torch_dispatch__`` sits above device dispatch, so it sees every op the model
routes to the device — the attention matmuls, the projection/MLP ``addmm``s,
RoPE, RMSNorm, sampling, etc. — regardless of which module issued it.

Debug-only. It is very verbose (hundreds of ops per forward pass) and adds a
Python-level hook to every op, so leave it off in normal runs.
"""

from __future__ import annotations

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

_TPU_DEVICE_TYPE = "tpu"


def _tensor_specs(flat_values) -> list[str]:
    """``[shape:dtype, ...]`` for each tensor in a flattened arg/result list."""
    specs = []
    for value in flat_values:
        if isinstance(value, torch.Tensor):
            dtype = str(value.dtype).replace("torch.", "")
            specs.append(f"{tuple(value.shape)}:{dtype}")
    return specs


def _touches_tpu(flat_values) -> bool:
    return any(
        isinstance(value, torch.Tensor) and value.device.type == _TPU_DEVICE_TYPE
        for value in flat_values
    )


class TpuOpTracer(TorchDispatchMode):
    """Print op type + input/output tensor shapes for every op hitting the TPU."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)

        in_flat, _ = tree_flatten((args, kwargs))
        out_flat, _ = tree_flatten(out)
        if _touches_tpu(in_flat) or _touches_tpu(out_flat):
            print(
                f"[TPU-OP] {func} "
                f"in={_tensor_specs(in_flat)} out={_tensor_specs(out_flat)}",
                flush=True,
            )
        return out


# Kept alive for the process lifetime once entered; a mode that is garbage
# collected while still active would leave the dispatch stack inconsistent.
_tracer: TpuOpTracer | None = None


def maybe_enable_tpu_op_trace() -> None:
    """Install the op tracer process-wide when ``SGLANG_DEBUG_TPU_TRACE`` is set.

    Idempotent: a second call is a no-op. Called from the TPU platform's
    ``init_backend`` so the mode is active for the whole model forward in
    whichever process (scheduler subprocess) runs it.
    """
    global _tracer
    from sglang.srt.environ import envs

    if _tracer is not None or not envs.SGLANG_DEBUG_TPU_TRACE.get():
        return
    _tracer = TpuOpTracer()
    _tracer.__enter__()
