"""D3: does history help, and does dropping it hurt? Train and val only, never test.

Written before running (docs/discussion_2026-09-21_next_steps.md section 1.3):
  * targets: the same frames D2 used (moving > 3 m/s, label not bridging a CAN gap), taken at a
    10 Hz base rate; history window 3 s, i.e. 30 steps ending at the target frame;
  * models: the D2 single-frame head, and an LSTM+dt (hidden 64) that sees the same window;
  * conditions share ONE checkpoint: 0 / 50 / 75 % of the HISTORY observations are dropped, the
    current frame always stays, and dt is recomputed over what is left;
  * H1 history helps: LSTM at 0% beats the single-frame head by >= 5% of macro MAE, all seeds;
  * H2 dropping hurts: LSTM at 75% is >= 3% worse than at 0%, all seeds;
  * budget: three learning rates at one seed, then three seeds at the winner, per model family.

    python hdd/d3_train.py --features ~/workspace/hdd/features/dinov2_s1 --split ~/workspace/hdd/split_a \
        --out ~/workspace/hdd/d3/dinov2
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics
import tracking

MOVING = 3.0


def refuse_inside_git(path):
    d = os.path.abspath(path if os.path.isdir(path) else os.path.dirname(path) or ".")
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            sys.exit(f"refusing to write HDD-derived output inside a git working tree ({d})")
        parent = os.path.dirname(d)
        if parent == d:
            return
        d = parent


class Sessions:
    """Per session: features at the base rate, their timestamps, labels, and the target indices."""

    def __init__(self, features, split_dir, which, base_hz, window_s, device):
        info = json.load(open(os.path.join(split_dir, "split.json")))
        self.items, self.index, self.device = [], [], device
        self.steps = int(round(base_hz * window_s))
        for s, meta in sorted(info["sessions"].items()):
            if meta["split"] != which:
                continue
            p = os.path.join(features, f"{s}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p)
            kept = z["kept"]
            stride = max(1, int(round(30.0 / base_hz)))       # the features are at ~30 Hz
            sel = np.arange(0, len(kept), stride)
            feats = z["feats"][sel]
            idx = kept[sel]
            t = z["t_cam"][idx]
            steer, speed = z["steer"][idx], z["speed_mps"][idx]
            ok = np.isfinite(steer) & np.isfinite(speed) & (speed > MOVING)
            # a target needs a full window behind it, and that window must be contiguous in time
            tgt = np.flatnonzero(ok)
            tgt = tgt[tgt >= self.steps - 1]
            span = t[tgt] - t[tgt - (self.steps - 1)]
            tgt = tgt[(span > 0) & (span < window_s * 2.5)]     # no long recording gap inside
            if not len(tgt):
                continue
            self.items.append(dict(session=s, feats=torch.from_numpy(feats), t=torch.from_numpy(t),
                                   y=torch.from_numpy(steer.astype(np.float32)), tgt=tgt))
            self.index += [(len(self.items) - 1, int(i)) for i in tgt]
        self.index = np.array(self.index)

    def labels(self):
        return np.concatenate([it["y"].numpy()[it["tgt"]] for it in self.items]).astype(np.float64)

    def sessions(self):
        return np.concatenate([[it["session"]] * len(it["tgt"]) for it in self.items])

    def batches(self, batch, shuffle, rng=None):
        """Yield (features (B, T, D), dt (B, T), y (B,)) with the window ending at the target."""
        order = np.arange(len(self.items))
        if shuffle:
            rng.shuffle(order)
        for i in order:
            it = self.items[i]
            f = it["feats"].to(self.device, non_blocking=True)
            t = it["t"].to(self.device)
            y = it["y"].to(self.device)
            tgt = torch.from_numpy(it["tgt"]).to(self.device)
            if shuffle:
                tgt = tgt[torch.randperm(len(tgt), device=self.device)]
            for j in range(0, len(tgt), batch):
                b = tgt[j:j + batch]
                idx = b[:, None] - torch.arange(self.steps - 1, -1, -1, device=self.device)[None]
                yield f[idx].float(), t[idx], y[b].float(), b


def drop_history(t, keep_rate, gen):
    """Keep the last column (the current frame) and a random `keep_rate` of the history.
    Returns a mask and the dt to the previous KEPT observation, zero-padded where masked."""
    B, T = t.shape
    mask = torch.rand(B, T, device=t.device, generator=gen) < keep_rate
    mask[:, -1] = True
    return mask


def run_lstm(model, f, t, mask):
    """Feed only the kept steps, in order, with dt to the previous kept step."""
    B, T, D = f.shape
    # rebuild each row as a packed sequence of kept steps (ragged); pad on the left with zeros
    keep = mask.float()
    n = keep.sum(1).long()
    order = torch.argsort(torch.where(mask, torch.arange(T, device=f.device).expand(B, T),
                                      torch.full((B, T), T + 1, device=f.device)), dim=1)
    idx = order[:, :int(n.max())]
    kept_f = torch.gather(f, 1, idx[:, :, None].expand(-1, -1, D))
    kept_t = torch.gather(t, 1, idx)
    valid = torch.gather(mask, 1, idx)
    dt = torch.zeros_like(kept_t)
    dt[:, 1:] = (kept_t[:, 1:] - kept_t[:, :-1]).clamp(min=0)
    return model(kept_f, dt.to(kept_f.dtype), valid)


class LSTMdt(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(dim, hidden)
        self.rnn = nn.LSTM(hidden + 1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, f, dt, valid):
        x = torch.cat([self.proj(self.norm(f)), dt[:, :, None]], -1) * valid[:, :, None]
        out, _ = self.rnn(x)
        last = valid.float().cumsum(1).argmax(1)                  # last kept step per row
        return self.head(out[torch.arange(len(out), device=out.device), last]).squeeze(-1)


class FrameMLP(nn.Module):
    def __init__(self, dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim, elementwise_affine=False), nn.Linear(dim, hidden),
                                 nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, f, dt, valid):
        return self.net(f[:, -1]).squeeze(-1)                      # the current frame only


def evaluate(model, data, mu, sd, keep_rate, seed):
    model.eval()
    gen = torch.Generator(device=data.device).manual_seed(seed)
    preds, ys = [], []
    with torch.no_grad():
        for f, t, y, _ in data.batches(512, shuffle=False):
            mask = drop_history(t, keep_rate, gen) if keep_rate < 1 else torch.ones_like(t, dtype=torch.bool)
            preds.append((run_lstm(model, f, t, mask) * sd + mu).cpu().numpy())
            ys.append(y.cpu().numpy())
    return np.concatenate(preds).astype(np.float64), np.concatenate(ys).astype(np.float64)


def train_once(kind, train, val, lr, seed, epochs, patience, batch, mu, sd, dim, run=tracking.Off()):
    torch.manual_seed(seed)
    model = (LSTMdt(dim) if kind == "lstm" else FrameMLP(dim)).to(train.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        loss_sum, n_batches = torch.zeros((), device=train.device), 0
        for f, t, y, _ in train.batches(batch, shuffle=True, rng=rng):
            mask = torch.ones_like(t, dtype=torch.bool)
            loss = ((run_lstm(model, f, t, mask) - (y - mu) / sd) ** 2).mean()
            loss_sum += loss.detach()
            n_batches += 1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        pred, y = evaluate(model, val, mu, sd, 1.0, seed)
        score = metrics.macro_mae(pred, y)
        if score < best - 1e-6:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        run.log({"val/macro_mae": score, "val/best_macro_mae": best, "train/patience_used": bad,
                 "train/loss": loss_sum.item() / max(n_batches, 1)}, step=ep + 1)
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model, best, ep + 1


def open_run(a, project, name, group, tags, **extra):
    """One SwanLab run per train_once call; these scripts keep no checkpoints, so none resume."""
    return tracking.start(a.swanlab, project, name, config=dict(vars(a), **extra), group=group, tags=tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-hz", type=float, default=10.0)
    ap.add_argument("--window-s", type=float, default=3.0)
    ap.add_argument("--keep-rates", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-4, 1e-4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    tracking.add_argument(ap)
    a = ap.parse_args()
    refuse_inside_git(a.out)
    os.makedirs(a.out, exist_ok=True)
    os.chmod(a.out, 0o700)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    train = Sessions(a.features, a.split, "train", a.base_hz, a.window_s, device)
    val = Sessions(a.features, a.split, "val", a.base_hz, a.window_s, device)
    ytr, yva = train.labels(), val.labels()
    mu, sd = float(ytr.mean()), float(ytr.std())
    dim = train.items[0]["feats"].shape[1]
    c_val = metrics.best_constant(yva)
    print(f"train {len(ytr):,} targets / {len(train.items)} sessions, val {len(yva):,} / {len(val.items)}; "
          f"{train.steps} steps at {a.base_hz:g} Hz ({a.window_s:g} s); loaded in {time.perf_counter()-t0:.0f}s")
    print(f"val constant {c_val:+.3f} deg -> macro MAE {metrics.macro_mae(np.full_like(yva, c_val), yva):.3f}")

    results = {}
    for kind in ("mlp", "lstm"):
        runs = []
        for lr in a.lrs:
            run = open_run(a, "CSC413-D3", f"{kind}_lrsearch_lr{lr:g}_s{a.seeds[0]}", f"{kind}_lr_search",
                           [kind, "lr_search", f"lr{lr:g}", f"seed{a.seeds[0]}"], kind=kind,
                           stage="lr_search", lr=lr, seed=a.seeds[0])
            _, score, eps = train_once(kind, train, val, lr, a.seeds[0], a.epochs, a.patience, a.batch, mu, sd, dim, run)
            run.finish()
            runs.append(dict(lr=lr, val_macro_mae=score, epochs=eps))
            print(f"  {kind} lr {lr:g}: val macro MAE {score:.3f} ({eps} epochs)")
        best_lr = min(runs, key=lambda r: r["val_macro_mae"])["lr"]
        finals = []
        for seed in a.seeds:
            run = open_run(a, "CSC413-D3", f"{kind}_final_lr{best_lr:g}_s{seed}", f"{kind}_final",
                           [kind, "final", f"lr{best_lr:g}", f"seed{seed}"], kind=kind, stage="final",
                           lr=best_lr, seed=seed)
            model, score, eps = train_once(kind, train, val, best_lr, seed, a.epochs, a.patience, a.batch, mu, sd, dim, run)
            row = dict(seed=seed, lr=best_lr, epochs=eps, conditions={})
            for keep in (a.keep_rates if kind == "lstm" else [1.0]):
                pred, y = evaluate(model, val, mu, sd, keep, seed)
                row["conditions"][f"keep_{keep:g}"] = dict(
                    macro_mae=metrics.macro_mae(pred, y), ccc=metrics.ccc(pred, y))
            run.log({f"final/keep{round(float(k.split('_', 1)[1]) * 100)}/{m}": v[m]
                     for k, v in row["conditions"].items() for m in ("macro_mae", "ccc")}, step=eps)
            run.finish()
            finals.append(row)
            cond = " ".join(f"{k} {v['macro_mae']:.3f}/{v['ccc']:+.2f}" for k, v in row["conditions"].items())
            print(f"  {kind} seed {seed} (lr {best_lr:g}, {eps} epochs): {cond}")
        results[kind] = dict(lr_search=runs, finals=finals)

    mlp = [f["conditions"]["keep_1"]["macro_mae"] for f in results["mlp"]["finals"]]
    full = [f["conditions"]["keep_1"]["macro_mae"] for f in results["lstm"]["finals"]]
    drop = [f["conditions"]["keep_0.25"]["macro_mae"] for f in results["lstm"]["finals"]]
    h1 = [(m - l) / m for m, l in zip(mlp, full)]
    h2 = [(d - l) / l for l, d in zip(full, drop)]
    gate = dict(h1_gain=[round(x, 4) for x in h1], h1_passed=all(x >= 0.05 for x in h1),
                h2_loss=[round(x, 4) for x in h2], h2_passed=all(x >= 0.03 for x in h2))
    gate["passed"] = gate["h1_passed"] and gate["h2_passed"]
    print(f"H1 history helps (>= 5%): {[f'{x:+.1%}' for x in h1]} -> {gate['h1_passed']}")
    print(f"H2 dropping 75% hurts (>= 3%): {[f'{x:+.1%}' for x in h2]} -> {gate['h2_passed']}")
    print(f"D3 {'PASS' if gate['passed'] else 'FAIL'} in {(time.perf_counter() - t0) / 60:.1f} min")
    out = dict(base_hz=a.base_hz, window_s=a.window_s, keep_rates=a.keep_rates,
               val_constant=c_val, results=results, gate=gate)
    p = os.path.join(a.out, "d3_results.json")
    json.dump(out, open(p, "w"), indent=1)
    os.chmod(p, 0o600)


if __name__ == "__main__":
    main()
