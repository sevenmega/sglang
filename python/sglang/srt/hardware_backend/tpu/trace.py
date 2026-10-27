"""Op-level tracer for the Sophgo SG2260E TPU device.

Enabled by ``SGLANG_DEBUG_TPU_TRACE=1``, and covers both halves of a forward
pass, which need different mechanisms:

* **aten ops** -- :class:`TpuOpTracer`, a process-wide :class:`TorchDispatchMode`
  that prints one ``[TPU-OP]`` line per aten op whose inputs or outputs live on
  the ``"tpu"`` device, with the op type and the shape (and dtype) of every
  tensor argument and result.  ``__torch_dispatch__`` sits above device
  dispatch, so it sees every op the model routes to the device -- the attention
  matmuls, the projection/MLP ``addmm``s, RoPE, RMSNorm, sampling, etc. --
  regardless of which module issued it.

* **tilelang kernels** -- :func:`trace_tpu_kernel`, a context manager wrapped
  around every launch.  A tilelang/PPL kernel is a Python call into the runtime,
  *not* an aten op, so the dispatch mode above structurally cannot see it; the
  two traces together are what distinguish "fused into a kernel" from "still an
  aten op".  Launches print ``[TPU-KERNEL]``.

Debug-only. Both are very verbose (hundreds of lines per forward pass) and add a
Python-level hook to every op, so leave them off in normal runs.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from sglang.srt.environ import envs

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
    if _tracer is not None or not envs.SGLANG_DEBUG_TPU_TRACE.get():
        return
    _tracer = TpuOpTracer()
    _tracer.__enter__()


@contextmanager
def trace_tpu_kernel(name: str, /, **fields) -> Iterator[None]:
    """Bracket one tilelang kernel launch with ``[TPU-KERNEL]`` lines.

    One line on entry carrying the kernel and the shape decisions the driver
    made (token count, padded row tile, ...), one on exit with the host time the
    call took.  The exit line is the *launch* cost, not the kernel's runtime:
    launches are asynchronous, so the device work continues after ``yield``
    returns.  Read it as "how long did the driver block", not "how fast was the
    kernel".

    A no-op when the flag is off, so the wrapper is free to sit on hot decode
    paths.
    """
    if not envs.SGLANG_DEBUG_TPU_TRACE.get():
        yield
        return
    spec = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[TPU-KERNEL] -> {name} {spec}".rstrip(), flush=True)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1e3
        print(f"[TPU-KERNEL] <- {name} launch={elapsed_ms:.3f}ms", flush=True)
