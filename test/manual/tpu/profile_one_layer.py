"""Top-down per-layer profile of Qwen3-0.6B prefill on the SG2260E TPU.

Drives *one* prefill forward with the same load path as
``python -m sglang.benchmark.one_batch`` (batch 32 x input 256), then times one
decoder layer end to end and each of its major ops in turn.

Every measurement is bracketed by a device synchronize, so the per-op numbers
are wall-clock *serialized* costs -- they sum to more than the unsynced layer
time whenever the runtime overlaps work. Both are reported so the gap (the
overlap) is visible rather than hidden.

Run:
    SGLANG_TPU_DEVICE_ID=5 python3 profile_one_layer.py --batch-size 32 --input-len 256
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch

# --- tuning knobs for the measurement itself -------------------------------
WARMUP_ROUNDS = 2
MEASURE_ROUNDS = 7
SYNC = None  # bound in main() once the device is up

# --- device peaks, as given -------------------------------------------------
# "fp16 16 TFLOPS": count a MAC as 2 FLOP, so the MAC ceiling is 8e12 MAC/s.
PEAK_TFLOPS = 16.0
PEAK_MAC_S = PEAK_TFLOPS * 1e12 / 2
PEAK_BW_B_S = 50e9  # DDR


def ceildiv(a: int, b: int) -> int:
    return -(-a // b)


def time_op(fn, rounds: int = MEASURE_ROUNDS) -> float:
    """Median serialized wall time of ``fn()`` in seconds (device synchronized)."""
    for _ in range(WARMUP_ROUNDS):
        fn()
    SYNC()
    samples = []
    for _ in range(rounds):
        SYNC()
        tic = time.perf_counter()
        fn()
        SYNC()
        samples.append(time.perf_counter() - tic)
    return statistics.median(samples)


class Op:
    """One timed unit: a name, a callable, and its MAC/byte accounting."""

    def __init__(self, name: str, fn, macs: float, bytes_moved: float, note: str = ""):
        self.name = name
        self.fn = fn
        self.macs = macs
        self.bytes_moved = bytes_moved
        self.note = note
        self.seconds: float | None = None

    def run(self) -> None:
        self.seconds = time_op(self.fn)

    @property
    def mac_rate(self) -> float:
        return self.macs / self.seconds if self.seconds else 0.0

    @property
    def mac_util(self) -> float:
        return 100.0 * self.mac_rate / PEAK_MAC_S

    @property
    def bw(self) -> float:
        return self.bytes_moved / self.seconds if self.seconds else 0.0

    @property
    def bw_util(self) -> float:
        return 100.0 * self.bw / PEAK_BW_B_S

    @property
    def arithmetic_intensity(self) -> float:
        # MAC per byte, the unit the roofline ridge point is expressed in.
        return self.macs / self.bytes_moved if self.bytes_moved else float("inf")


def build_layer_ops(layer, hidden_states, positions, forward_batch, cfg, ctx) -> list[Op]:
    """Wrap each major op of one decoder layer with its MAC / byte accounting.

    Shapes (Qwen3-0.6B, prefill T tokens total):
        H  = 1024   hidden
        I  = 3072   intermediate
        Hq = 16, Hkv = 8, D = 128
    Weight bytes use 2 B/elem (fp16); activation bytes count every tensor read
    or written by the op, which is what actually crosses DDR.
    """
    T = hidden_states.shape[0]
    H = cfg.hidden_size
    I = cfg.intermediate_size
    Hq = cfg.num_attention_heads
    Hkv = cfg.num_key_value_heads
    D = cfg.head_dim
    q_dim = Hq * D
    kv_dim = Hkv * D
    bpe = 2  # fp16/bf16 bytes per element

    attn = layer.self_attn
    mlp = layer.mlp
    norm = layer.input_layernorm
    post_norm = layer.post_attention_layernorm

    # --- accounting ---------------------------------------------------------
    # qkv: T x H  ->  T x (q_dim + 2*kv_dim)
    qkv_macs = T * H * (q_dim + 2 * kv_dim)
    qkv_bytes = (H * (q_dim + 2 * kv_dim) + T * H + T * (q_dim + 2 * kv_dim)) * bpe

    # Attention. The kernel runs one launch per bucketed request set with a
    # block-diagonal mask, so a query attends ONLY its own request's keys --
    # NOT all T. Batch B requests x S tokens each => B * S^2 score pairs,
    # i.e. T*S with T = B*S, not T*T. Padded to block_M/block_N multiples.
    B_req = forward_batch.batch_size
    S = ceildiv(T // B_req, 8) * 8          # block_M padding
    S_kv = ceildiv(S, 128) * 128            # block_N padding, floored at S
    attn_macs = B_req * S * S_kv * Hq * D * 2  # QK^T then P@V
    attn_bytes = (
        B_req * S * q_dim           # q (padded)
        + 2 * B_req * S_kv * kv_dim  # k, v (padded)
        + B_req * S * S_kv           # mask build + mask read
        + B_req * S * q_dim          # out
        + 2 * B_req * S_kv * kv_dim  # kv cache write
    ) * bpe
    print(
        f"[prof] attn accounting: B_req={B_req} S={S} S_kv={S_kv} "
        f"Hq={Hq} D={D} -> {attn_macs/1e6:.1f} MMAC "
        f"(naive T*T would be {T*T*Hq*D*2/1e6:.1f})",
        flush=True,
    )

    # o_proj: T x q_dim -> T x H
    o_macs = T * q_dim * H
    o_bytes = (q_dim * H + T * q_dim + T * H) * bpe

    # MLP gate+up: T x H -> T x 2I
    gu_macs = T * H * 2 * I
    gu_bytes = (H * 2 * I + T * H + T * 2 * I) * bpe

    # MLP down: T x I -> T x H
    down_macs = T * I * H
    down_bytes = (I * H + T * I + T * H) * bpe

    # RMSNorm: elementwise, ~0 MAC, but a full read+write of the activation.
    norm_macs = T * H
    norm_bytes = 2 * T * H * 4  # fp32 stats traffic dominates a norm

    # --- callables ----------------------------------------------------------
    def run_qkv():
        return attn.qkv_proj(hidden_states)

    qkv, _ = run_qkv()
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)

    def run_attn():
        from sglang.srt.model_executor.forward_context import forward_context
        with forward_context(ctx):
            return attn.attn(q, k, v, forward_batch, save_kv_cache=True)

    attn_out = run_attn()

    def run_oproj():
        out, _ = attn.o_proj(attn_out)
        return out

    o_out = run_oproj()

    def run_gate_up():
        gu, _ = mlp.gate_up_proj(o_out)
        return gu

    gate_up = run_gate_up()

    def run_act():
        return mlp.act_fn(gate_up)

    act_out = run_act()

    def run_down():
        d, _ = mlp.down_proj(act_out)
        return d

    def run_norm():
        return norm(hidden_states)

    return [
        Op("input_layernorm", run_norm, norm_macs, norm_bytes),
        Op("qkv_proj", run_qkv, qkv_macs, qkv_bytes),
        Op("attn(flash)", run_attn, attn_macs, attn_bytes, "causal; S_kv=T"),
        Op("o_proj", run_oproj, o_macs, o_bytes),
        Op("post_attn_norm", run_norm, norm_macs, norm_bytes),
        Op("gate_up_proj", run_gate_up, gu_macs, gu_bytes),
        Op("silu_mul", run_act, T * I, 3 * T * I * bpe),
        Op("down_proj", run_down, down_macs, down_bytes),
    ]


def main():
    global SYNC
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

    sys.argv = [
        sys.argv[0],
        "--model-path", args.model_path,
        "--base-gpu-id", DEV,
    ]
    import argparse as _ap

    from sglang.benchmark.one_batch import (
        _maybe_prepare_mlp_sync_batch,
        load_model,
        prepare_synthetic_inputs_for_latency_test,
    )
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    from types import SimpleNamespace

    suppress_other_loggers()

    # Build ServerArgs the same way the CLI does.
    parser = _ap.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    sa_args = parser.parse_args(
        ["--model-path", args.model_path, "--base-gpu-id", DEV]
    )
    server_args = ServerArgs.from_cli_args(sa_args)
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)

    print(f"[prof] loading model from {args.model_path} ...", flush=True)
    bench_runner, tokenizer = load_model(server_args, port_args, args.device_id, 0)
    # ``load_model`` returns a bench wrapper; the real ModelRunner is inside.
    model_runner = bench_runner.torch_runner

    dev = model_runner.device
    SYNC = lambda: torch.get_device_module(dev).synchronize()
    print(f"[prof] device={dev}", flush=True)

    # --- build the prefill batch -------------------------------------------
    reqs = prepare_synthetic_inputs_for_latency_test(args.batch_size, args.input_len)
    bench_runner.clear()

    class _TreeCache(SimpleNamespace):
        def supports_swa(self): return False
        def supports_mamba(self): return False
        def is_chunk_cache(self): return False
        def is_tree_cache(self): return not self.is_chunk_cache()
        def evict(self, params): pass

    tree = _TreeCache(
        page_size=server_args.page_size,
        device=dev,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
    )
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=tree,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    if batch.input_ids is None and getattr(batch, "prefill_input_ids_cpu", None) is not None:
        batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
        batch.prefill_input_ids_cpu = None
    fwd = ForwardBatch.init_new(batch, model_runner, return_hidden_states_before_norm=False)
    print(f"[prof] prefill tokens T = {fwd.input_ids.shape[0]}", flush=True)

    model = model_runner.model
    layers = model.model.layers if hasattr(model, "model") else model.layers
    cfg = model.config if hasattr(model, "config") else model_runner.model_config.hf_config
    print(f"[prof] layers = {len(layers)}", flush=True)

    # The attention backend is read from an ambient forward context that
    # ModelRunner._forward_raw installs; a bare layer call needs it too.
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

    ctx = ForwardContext(attn_backend=model_runner.attn_backend)

    # --- warm the whole forward once so caches/weights are resident ---------
    with torch.no_grad():
        model_runner.forward(fwd)
    SYNC()

    # --- measure one full layer, unsynchronized (true e2e) ------------------
    # Feed the layer its real inputs from the live forward state.
    layer0 = layers[0]
    with torch.no_grad():
        probe = model.model.embed_tokens(fwd.input_ids) if hasattr(model, "model") else None
    if probe is None:
        probe = model.embed_tokens(fwd.input_ids)
    positions = fwd.positions if hasattr(fwd, "positions") and fwd.positions is not None else None
    if positions is None:
        import torch as _t
        positions = _t.arange(probe.shape[0], device=dev, dtype=torch.int64)

    def full_layer():
        with torch.no_grad(), forward_context(ctx):
            return layer0(positions, probe, fwd, residual=None)

    e2e_s = time_op(full_layer)
    print(f"\n[prof] one decoder layer, e2e (serialized): {e2e_s*1e6:.1f} us", flush=True)

    # --- per-op breakdown ---------------------------------------------------
    ops = build_layer_ops(layer0, probe, positions, fwd, cfg, ctx)
    for op in ops:
        op.run()

    ridge = PEAK_MAC_S / PEAK_BW_B_S  # MAC/byte at which compute and BW balance
    print(f"\n[prof] roofline ridge point: {ridge:.1f} MAC/byte "
          f"(peak {PEAK_MAC_S/1e12:.1f} T-MAC/s, {PEAK_BW_B_S/1e9:.0f} GB/s)")
    print()
    hdr = f"{'op':<17}{'time(us)':>10}{'MAC':>12}{'T-MAC/s':>10}{'MAC%':>8}{'GB/s':>9}{'BW%':>7}{'AI':>8}  bound"
    print(hdr)
    print("-" * len(hdr))

    total_t = 0.0
    total_macs = 0.0
    total_bytes = 0.0
    rows = []
    for op in ops:
        total_t += op.seconds
        total_macs += op.macs
        total_bytes += op.bytes_moved
        bound = "compute" if op.arithmetic_intensity > ridge else "memory"
        if op.macs < 1e5:
            bound = "memory"
        rows.append((op, bound))
        print(
            f"{op.name:<17}{op.seconds*1e6:>10.1f}{op.macs/1e6:>12.1f}"
            f"{op.mac_rate/1e12:>10.4f}{op.mac_util:>8.2f}{op.bw/1e9:>9.2f}"
            f"{op.bw_util:>7.2f}{op.arithmetic_intensity:>8.1f}  {bound}"
        )
    print("-" * len(hdr))
    print(
        f"{'SUM(ops)':<17}{total_t*1e6:>10.1f}{total_macs/1e6:>12.1f}"
        f"{total_macs/total_t/1e12:>10.4f}{100*total_macs/total_t/PEAK_MAC_S:>8.2f}"
        f"{total_bytes/total_t/1e9:>9.2f}{100*total_bytes/total_t/PEAK_BW_B_S:>7.2f}"
    )
    print(
        f"\n[prof] sum(ops) = {total_t*1e6:.1f} us vs e2e = {e2e_s*1e6:.1f} us "
        f"-> overlap/launch gap = {(total_t-e2e_s)*1e6:+.1f} us"
    )

    # --- whole-model extrapolation -----------------------------------------
    n_layers = len(layers)
    prefill_lat = e2e_s * n_layers
    print(
        f"\n[prof] extrapolated prefill ({n_layers} layers x {e2e_s*1e6:.1f} us) "
        f"= {prefill_lat*1e3:.2f} ms; measured run was ~7.6 s"
    )


if __name__ == "__main__":
    main()
