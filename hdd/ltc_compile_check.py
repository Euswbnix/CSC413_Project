"""Is the torch.compile'd LTC cell numerically the same as ncps's eager one under real training
conditions, and does it stay compiled?

Real training mixes batch sizes (every session ends on its own remainder), switches between
training and no-grad evaluation, and feeds a first hidden state that needs no gradient followed by
ones that do. Each of those can trigger a recompilation; past torch's limit it silently falls back
to eager. This check exercises all of them, compares outputs and every parameter gradient, reports
how many graphs were compiled, and times both paths.

    python hdd/ltc_compile_check.py --cache ~/data/hdd/cache/dinov2_10hz --ckpt <ltc checkpoint>
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d4_run as D


def train_step(model, f, t, keep, seed, mu, sd, y):
    gen = torch.Generator(device=f.device).manual_seed(seed)
    out = D.forward_masked(model, f, t, keep, gen)
    loss = ((out - (y - mu) / sd) ** 2).mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    return out.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def eval_step(model, f, t, keep, seed):
    gen = torch.Generator(device=f.device).manual_seed(seed)
    with torch.no_grad():
        return D.forward_masked(model, f, t, keep, gen)


def rel(a, b):
    return float((a - b).abs().max() / (a.abs().max() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--unfolds", type=int, default=24)
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 173, 64, 211, 256, 97])
    ap.add_argument("--reps", type=int, default=12)
    a = ap.parse_args()
    dev = "cuda"
    data = D.Cached(a.cache, "train", 30, dev)
    y_all = data.labels()
    mu, sd = float(y_all.mean()), float(y_all.std())
    rng = np.random.default_rng(0)
    it = data.batches(256, shuffle=True, rng=rng)
    batches = []
    for n in a.sizes:
        f, t, y = next(it)
        batches.append((f[:n], t[:n], y[:n]))

    eager = D.Arm("ltc", batches[0][0].shape[2], seed=0, ode_unfolds=a.unfolds).to(dev)
    if a.ckpt:
        c = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        eager.load_state_dict(c["best_state"] or c["model"])
    comp = D.Arm("ltc", batches[0][0].shape[2], seed=0, ode_unfolds=a.unfolds, compile_ltc=True).to(dev)
    comp.load_state_dict(eager.state_dict())

    import torch._dynamo.utils as du
    du.counters.clear()
    worst_o = worst_g = worst_e = 0.0
    for i, (f, t, y) in enumerate(batches):
        keep = [1.0, 0.5, 0.25][i % 3]
        oe, ge = train_step(eager, f, t, keep, i, mu, sd, y)
        oc, gc = train_step(comp, f, t, keep, i, mu, sd, y)
        assert ge.keys() == gc.keys()
        worst_o = max(worst_o, rel(oe, oc))
        worst_g = max(worst_g, max(rel(ge[n], gc[n]) for n in ge))
        worst_e = max(worst_e, rel(eval_step(eager, f, t, keep, 50 + i), eval_step(comp, f, t, keep, 50 + i)))
    graphs = du.counters["stats"]["unique_graphs"]
    print(f"batch sizes {a.sizes}, train and eval interleaved")
    print(f"relative max difference: train outputs {worst_o:.2e}, gradients {worst_g:.2e}, eval outputs {worst_e:.2e}")
    print(f"compiled graphs: {graphs} (limit 64); recompilations beyond the first: {max(0, graphs - 1)}")

    for name, model in (("eager", eager), ("compiled", comp)):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for r in range(a.reps):
            f, t, y = batches[r % len(batches)]
            train_step(model, f, t, [1.0, 0.5, 0.25][r % 3], 100 + r, mu, sd, y)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / a.reps
        print(f"{name:>8}: {dt * 1000:7.1f} ms per training step over mixed sizes, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"graphs after timing: {du.counters['stats']['unique_graphs']} (should not have grown)")


if __name__ == "__main__":
    main()
