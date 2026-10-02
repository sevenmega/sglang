"""Decompose the TPU attention op: where do the 227ms/launch actually go?

Times each phase of the batched flash-attention driver separately:
  gather  -- slicing q/k/v out of the packed query + paged KV cache
  pack    -- host zero-pad into [B,H,S_pad,D] fp16
  mask    -- building the [B,q_pad,kv_pad] additive mask
  launch  -- the PPL kernel call (includes _sync_rt_device)
  scatter -- slicing the output back into [T,H,D]

Run:
    SGLANG_TPU_DEVICE_ID=6 python3 profile_attn_phases.py --batch-size 32 --input-len 256
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch

WARMUP = 2
ROUNDS = 5


def ceildiv(a, b):
    return -(-a // b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/Qwen3-0.6B")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--input-len", type=int, default=256)
    ap.add_argument("--device-id", type=int, default=6)
    args = ap.parse_args()
    DEV = str(args.device_id)

    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.utils import suppress_other_loggers
    import argparse as _ap
    from types import SimpleNamespace

    from sglang.benchmark.one_batch import (
        _maybe_prepare_mlp_sync_batch,
        load_model,
        prepare_synthetic_inputs_for_latency_test,
    )
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    suppress_other_loggers()
    parser = _ap.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    server_args = ServerArgs.from_cli_args(
        parser.parse_args(["--model-path", args.model_path, "--base-gpu-id", DEV])
    )
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)

    bench_runner, _ = load_model(server_args, port_args, args.device_id, 0)
    mr = bench_runner.torch_runner
    dev = mr.device
    SYNC = lambda: torch.get_device_module(dev).synchronize()
    print(f"[ph] device={dev}", flush=True)

    reqs = prepare_synthetic_inputs_for_latency_test(args.batch_size, args.input_len)
    bench_runner.clear()

    class _TreeCache(SimpleNamespace):
        def supports_swa(self): return False
        def supports_mamba(self): return False
        def is_chunk_cache(self): return False
        def is_tree_cache(self): return not self.is_chunk_cache()
        def evict(self, params): pass

    tree = _TreeCache(
        page_size=server_args.page_size, device=dev,
        token_to_kv_pool_allocator=mr.token_to_kv_pool_allocator,
    )
    batch = ScheduleBatch.init_new(
        reqs=reqs, req_to_token_pool=mr.req_to_token_pool,
        token_to_kv_pool_allocator=mr.token_to_kv_pool_allocator,
        tree_cache=tree, model_config=mr.model_config,
        enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, mr)
    if batch.input_ids is None and getattr(batch, "prefill_input_ids_cpu", None) is not None:
        batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
        batch.prefill_input_ids_cpu = None
    fwd = ForwardBatch.init_new(batch, mr, return_hidden_states_before_norm=False)
    ctx = ForwardContext(attn_backend=mr.attn_backend)

    model = mr.model
    layers = model.model.layers if hasattr(model, "model") else model.layers
    layer0 = layers[0]
    attn = layer0.self_attn

    with torch.no_grad(), forward_context(ctx):
        mr.forward(fwd)
    SYNC()

    # Recreate the layer-0 attention inputs.
    hs = model.model.embed_tokens(fwd.input_ids) if hasattr(model, "model") else model.embed_tokens(fwd.input_ids)
    positions = fwd.positions if fwd.positions is not None else torch.arange(hs.shape[0], device=dev, dtype=torch.int64)
    with torch.no_grad(), forward_context(ctx):
        normed = layer0.input_layernorm(hs)
        qkv, _ = attn.qkv_proj(normed)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)

    T = q.shape[0]
    Hq, D = attn.num_heads, attn.head_dim
    Hkv = attn.num_kv_heads
    print(f"[ph] T={T} Hq={Hq} Hkv={Hkv} D={D} batch={fwd.batch_size}", flush=True)

    # --- instrument the driver's phases ------------------------------------
    from sglang.srt.hardware_backend.tpu import attention as attn_mod

    backend = mr.attn_backend
    print(f"[ph] backend={type(backend).__name__}", flush=True)

    # Wrap the internal helpers to time them individually.
    phases: dict[str, list[float]] = {}

    def timed(name, fn):
        def wrapper(*a, **kw):
            SYNC()
            t0 = time.perf_counter()
            r = fn(*a, **kw)
            SYNC()
            phases.setdefault(name, []).append(time.perf_counter() - t0)
            return r
        return wrapper

    cls = type(backend)
    orig_gather = backend._gather_plan_qkv
    orig_mask_fn = cls._build_batched_additive_mask
    orig_flash_fn = cls._attend_flash_batch
    orig_scatter_fn = cls._scatter_plan_output

    backend._gather_plan_qkv = timed("gather", orig_gather)
    cls._build_batched_additive_mask = staticmethod(timed("mask", orig_mask_fn))
    cls._attend_flash_batch = staticmethod(timed("launch", orig_flash_fn))
    cls._scatter_plan_output = staticmethod(timed("scatter", orig_scatter_fn))

    pack_calls = []
    # Only the device packer exists now: the host-side ``_pack_flash_batch`` was
    # removed once the kernel was built in the model's own dtype (bf16), which
    # made the conversion it performed unnecessary.
    orig_pack_dev = attn_mod._pack_flash_batch_device

    def _timed(fn):
        def wrapper(*a, **kw):
            SYNC()
            t0 = time.perf_counter()
            r = fn(*a, **kw)
            SYNC()
            pack_calls.append(time.perf_counter() - t0)
            return r
        return wrapper

    attn_mod._pack_flash_batch_device = _timed(orig_pack_dev)

    def run_attn():
        phases.clear()
        pack_calls.clear()
        with torch.no_grad(), forward_context(ctx):
            return attn.attn(q, k, v, fwd, save_kv_cache=True)

    for _ in range(WARMUP):
        run_attn()
    SYNC()

    totals = []
    for _ in range(ROUNDS):
        SYNC()
        t0 = time.perf_counter()
        run_attn()
        SYNC()
        totals.append(time.perf_counter() - t0)

    total = statistics.median(totals)
    print(f"\n[ph] attn op total: {total*1e6:.0f} us  (median of {ROUNDS})")
    print(f"\n{'phase':<10}{'sum(us)':>10}{'calls':>7}{'%of total':>11}")
    print("-" * 38)
    acc = 0.0
    for name in ["mask", "gather", "pack", "launch", "scatter"]:
        vals = phases.get(name, [])
        if name == "pack":
            vals = pack_calls
        s = sum(vals)
        acc += s
        print(f"{name:<10}{s*1e6:>10.0f}{len(vals):>7}{100*s/total:>11.1f}")
    print("-" * 38)
    print(f"{'SUM':<10}{acc*1e6:>10.0f}{'':>7}{100*acc/total:>11.1f}")
    print(f"{'unaccounted':<10}{(total-acc)*1e6:>10.0f}{'':>7}{100*(total-acc)/total:>11.1f}")


if __name__ == "__main__":
    main()
