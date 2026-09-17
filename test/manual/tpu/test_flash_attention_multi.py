import os, sys, torch, torch_tpu
sys.path.insert(0, "/workspace/sglang/python")
sys.path.insert(0, "/workspace/tilelang")
dev = int(os.environ.get("TPU_VISIBLE_DEVICES","0").split(",")[0])
torch.tpu.set_device(dev)
from sglang.srt.hardware_backend.tpu.attention import TpuAttnBackend as B
Hq, Hkv, D = 16, 8, 128
# Mix of extend-like and decode-like shapes, repeated, plus repeats of the same
# shape (the case that failed even before the shape theory was disproved).
SHAPES = [(1,1),(1,8),(1,64),(3,97),(8,301),(1,512),(5,5),(12,140),
          (1,1),(1,300),(16,16),(8,301),(1,1),(7,1000),(2,33)]
fails = 0
for i, (sq, skv) in enumerate(SHAPES):
    q = torch.randn(Hq, sq, D, dtype=torch.float16, device=f"tpu:{dev}")
    k = torch.randn(Hkv, skv, D, dtype=torch.float16, device=f"tpu:{dev}")
    v = torch.randn(Hkv, skv, D, dtype=torch.float16, device=f"tpu:{dev}")
    out = torch.empty_like(q)
    m = B._build_additive_mask(q_len=sq, kv_len=skv, causal=True, sliding_window_size=None)
    r = B._attend_flash(q=q, k=k, v=v, out=out, scaling=D**-0.5, cpu_mask=m)
    g = Hq // Hkv
    qc, kc, vc = q.float().cpu(), k.float().cpu(), v.float().cpu()
    ref = torch.empty(Hq, sq, D)
    for h in range(Hq):
        ref[h] = torch.softmax((qc[h] @ kc[h//g].t()) * (D**-0.5) + m, -1) @ vc[h//g]
    rel = (out.float().cpu() - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
    bad = (not r) or rel >= 0.01
    fails += bad
    print(f"  [{i:2d}] sq={sq:4d} skv={skv:4d}  ret={r}  rel={rel:.2e}  {'ok' if not bad else 'FAIL'}", flush=True)
print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'} ({len(SHAPES)} launches, one process)")
