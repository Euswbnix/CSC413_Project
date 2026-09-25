"""Compute cost of the three recurrent arms, as fixed by docs/preregistration_2026-09-22.md 9.2.

Main table: one implementation style for all three -- eager, one Python loop over time steps that
calls a cell (nn.LSTMCell with the trained nn.LSTM's weights copied in, the repo's CfCCell,
ncps' LTCCell at 24 sub-steps), no torch.compile. Every step function is first checked against
the arm's own inference path on the same validation windows and masks (standardised outputs,
rtol 1e-4, atol 1e-5); an arm that fails is not timed. Only the temporal block is timed: its
input is the projected 64-d features and dt already on the GPU; encoder, projection, readout,
disk, mask drawing and host transfers are all outside the clock. State starts from zero for
every window.

Fixed conditions: one GPU, float32, TF32 off for both matmul and cuDNN, eval() and
inference_mode(); full history, 50% kept and 25% kept, with the same windows and masks for every
arm (real length differences kept; the padded execution length is reported). batch=1 for latency,
batch=256 for throughput and peak allocated memory. Each shape: 50 warm-up calls, then 3 rounds of
200, each call timed on the wall clock between synchronisations; median, p95 and windows/s.

A second table gives the engineering paths actually used in training (cuDNN nn.LSTM, torch.compile
of the CfC and LTC cells). It is not ranked against the main table.

    python hdd/d4_cost.py --cache cache/dinov2_10hz --ckpt-dir d4/dinov2 --out d4/analysis/cost_seed0.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import d4_run as D

ARMS = ("lstm", "cfc", "ltc")
KEEPS = (1.0, 0.5, 0.25)
RTOL, ATOL = 1e-4, 1e-5
WARMUP, CALLS, ROUNDS = 50, 200, 3


def other_gpu_processes():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    except OSError:
        return []
    rows = [line.split(",") for line in out.strip().splitlines() if line.strip()]
    return [dict(pid=int(p), mib=int(m)) for p, m in rows if int(p) != os.getpid()]


def pick_windows(data, n, seed=0):
    """n validation windows spread over all sessions, fixed by `seed`; returns f (n,T,dim), t (n,T)."""
    sizes = [len(it["tgt"]) for it in data.items]
    picks = np.sort(np.random.default_rng(seed).choice(sum(sizes), n, replace=False))
    starts = np.cumsum([0] + sizes[:-1])
    fs, ts = [], []
    for g in picks:
        i = int(np.searchsorted(starts, g, side="right") - 1)
        it, tgt = data.items[i], data.items[i]["tgt"][g - starts[i]]
        win = tgt - np.arange(data.steps - 1, -1, -1)
        fs.append(np.asarray(it["feats"][win], dtype=np.float32))
        ts.append(it["t"][win])
    return (torch.from_numpy(np.stack(fs)).to(data.device),
            torch.from_numpy(np.stack(ts)).to(data.device))


def pack(f, t, mask):
    """forward_masked's packing for an explicit mask: kept frames first, in time order."""
    T = t.shape[1]
    order = torch.argsort(torch.where(mask, torch.arange(T, device=t.device).expand_as(t),
                                      torch.full_like(t, T + 1, dtype=torch.long)), dim=1)
    n = int(mask.sum(1).max())
    idx = order[:, :n]
    kf = torch.gather(f, 1, idx[:, :, None].expand(-1, -1, f.shape[2]))
    kt = torch.gather(t, 1, idx)
    valid = torch.gather(mask, 1, idx)
    dt = torch.zeros_like(kt)
    dt[:, 1:] = (kt[:, 1:] - kt[:, :-1]).clamp(min=0)
    return kf, dt.to(kf.dtype), valid, (kt - t[:, -1:]).to(kf.dtype)


class Stepwise(nn.Module):
    """The shared outer loop: zero state, one cell call per time step, every step's output kept
    (as the original blocks return them), for any of the three cells."""

    def __init__(self, arm):
        super().__init__()
        self.kind = arm.kind
        if arm.kind == "lstm":
            src = arm.rnn
            self.cell = nn.LSTMCell(src.input_size, src.hidden_size).to(src.weight_ih_l0.device)
            with torch.no_grad():
                for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    getattr(self.cell, name).copy_(getattr(src, f"{name}_l0"))
        else:
            self.cell = arm.rnn.cell                    # CfCCell, or the eager ncps LTCCell
        self.hidden = arm.proj.out_features

    def forward(self, x, dt):
        B = x.shape[0]
        h = x.new_zeros(B, self.hidden)
        c = x.new_zeros(B, self.hidden)
        out = []
        for i in range(x.shape[1]):
            if self.kind == "lstm":
                h, c = self.cell(torch.cat([x[:, i], dt[:, i:i + 1]], -1), (h, c))
            elif self.kind == "cfc":
                h = self.cell(x[:, i], h, dt[:, i:i + 1])
            else:
                h, _ = self.cell(x[:, i].contiguous(), h, dt[:, i:i + 1].clamp(min=D.MIN_DT).contiguous())
            out.append(h)
        return torch.stack(out, 1)


def engineering_block(arm):
    """The path each arm actually trained with: cuDNN LSTM, compiled CfC cell, compiled LTC cell."""
    if arm.kind == "lstm":
        return lambda x, dt: arm.rnn(torch.cat([x, dt[:, :, None]], -1))[0]
    import torch._dynamo as dynamo            # an alias: a bare `import torch._dynamo` here would
    dynamo.config.recompile_limit = 64        # make `torch` local to this function (as in training)
    step = torch.compile(arm.rnn.cell, dynamic=True)

    def run(x, dt):
        h = x.new_zeros(x.shape[0], arm.proj.out_features)
        out = []
        for i in range(x.shape[1]):
            if arm.kind == "cfc":
                h = step(x[:, i].contiguous(), h, dt[:, i:i + 1].contiguous())
            else:
                h, _ = step(x[:, i].contiguous(), h, dt[:, i:i + 1].clamp(min=D.MIN_DT).contiguous())
            out.append(h)
        return torch.stack(out, 1)
    return run


def readout(arm, states, valid):
    last = valid.float().cumsum(1).argmax(1)
    return arm.head(states[torch.arange(len(states), device=states.device), last]).squeeze(-1)


def time_calls(fn, warmup=WARMUP, calls=CALLS, n_rounds=ROUNDS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    rounds = []
    for _ in range(n_rounds):
        ms = []
        for _ in range(calls):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t0) * 1e3)
        rounds.append(ms)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    allms = np.concatenate(rounds)
    return dict(median_ms=float(np.median(allms)), p95_ms=float(np.percentile(allms, 95)),
                round_medians_ms=[float(np.median(r)) for r in rounds],
                peak_alloc_mib=peak / 2 ** 20, peak_above_inputs_mib=(peak - base) / 2 ** 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt-dir", required=True, help="where <arm>_lr_chosen.json and ckpt/ live")
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--allow-busy-gpu", action="store_true")
    ap.add_argument("--skip-engineering", action="store_true")
    ap.add_argument("--warmup", type=int, default=WARMUP, help="registered: 50 (lower only to smoke-test)")
    ap.add_argument("--calls", type=int, default=CALLS, help="registered: 200")
    ap.add_argument("--rounds", type=int, default=ROUNDS, help="registered: 3")
    a = ap.parse_args()
    reps = dict(warmup=a.warmup, calls=a.calls, n_rounds=a.rounds)
    D.refuse_inside_git(a.out)
    assert torch.cuda.is_available(), "timing needs the GPU"
    busy = other_gpu_processes()
    if busy and not a.allow_busy_gpu:
        sys.exit(f"other processes are on the GPU {busy}; the registration requires it idle")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    val = D.Cached(a.cache, "val", 30, dev)
    f, t = pick_windows(val, a.windows)
    gen = torch.Generator(device=dev).manual_seed(7)
    masks = {keep: D.masked(t, keep, gen)[0] for keep in KEEPS}          # shared by every arm
    res = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
               cudnn=torch.backends.cudnn.version(), tf32=dict(matmul=False, cudnn=False),
               precision="float32", windows=a.windows, warmup=a.warmup, calls=a.calls, rounds=a.rounds,
               gpu_processes_at_start=busy, check=dict(rtol=RTOL, atol=ATOL), arms={})
    for kind in ARMS:
        lr = json.load(open(os.path.join(a.ckpt_dir, f"{kind}_lr_chosen.json")))["chosen_lr"]
        name = f"{kind}_lr{lr:g}_s0{'_u24' if kind == 'ltc' else ''}.pt"
        state = torch.load(os.path.join(a.ckpt_dir, "ckpt", name), map_location="cpu",
                           weights_only=False)["best_state"]
        arm = D.Arm(kind, f.shape[2], seed=0, ode_unfolds=24).to(dev).eval()
        arm.load_state_dict(state)
        step = Stepwise(arm).eval()
        eng = None if a.skip_engineering else engineering_block(arm)
        params = sum(p.numel() for p in arm.rnn.parameters() if p.requires_grad)
        row = dict(checkpoint=name, temporal_params=params, conditions={})
        with torch.inference_mode():
            for keep in KEEPS:
                cond = {}
                for B in (1, a.windows):
                    kf, dt, valid, rel = pack(f[:B], t[:B], masks[keep][:B])
                    x = arm.proj(arm.norm(kf)) * valid[:, :, None]           # outside the clock
                    ref = arm(kf, dt, valid, rel)                            # the arm's own path
                    got = readout(arm, step(x, dt), valid)
                    err = float((got - ref).abs().max())
                    ok = bool(torch.allclose(got, ref, rtol=RTOL, atol=ATOL))
                    entry = dict(exec_length=int(x.shape[1]),
                                 mean_kept=float(valid.sum(1).float().mean()),
                                 check_max_abs_err=err, check_passed=ok)
                    if ok:
                        entry["eager"] = time_calls(lambda: step(x, dt), **reps)
                        entry["eager"]["windows_per_s"] = B / (entry["eager"]["median_ms"] / 1e3)
                    if eng is not None:
                        e_got = readout(arm, eng(x, dt), valid)
                        e_ok = bool(torch.allclose(e_got, ref, rtol=RTOL, atol=ATOL))
                        entry["engineering_check_max_abs_err"] = float((e_got - ref).abs().max())
                        entry["engineering_check_passed"] = e_ok
                        entry["engineering"] = time_calls(lambda: eng(x, dt), **reps)
                        entry["engineering"]["windows_per_s"] = B / (entry["engineering"]["median_ms"] / 1e3)
                    cond[f"batch_{B}"] = entry
                    e = entry.get("eager", {})
                    print(f"{kind:4s} keep {keep:4g} batch {B:3d} len {entry['exec_length']:2d}: check "
                          f"{'ok' if ok else 'FAILED'} ({err:.1e})  eager median {e.get('median_ms', float('nan')):8.3f} ms "
                          f"p95 {e.get('p95_ms', float('nan')):8.3f}  {e.get('windows_per_s', float('nan')):10.1f} win/s  "
                          f"peak {e.get('peak_alloc_mib', float('nan')):7.1f} MiB", flush=True)
                row["conditions"][f"keep_{keep:g}"] = cond
        res["arms"][kind] = row
        del arm, step, eng
        torch.cuda.empty_cache()
    res["gpu_processes_at_end"] = other_gpu_processes()
    if res["gpu_processes_at_end"]:
        print(f"WARNING: other GPU processes appeared during timing: {res['gpu_processes_at_end']}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    os.chmod(a.out, 0o600)


if __name__ == "__main__":
    main()
