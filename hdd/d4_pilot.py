"""D4 pilot on the development split: how big is the CfC-minus-LSTM difference, and how much does
it wobble across seeds, route clusters and drop masks?

This is the run the pre-registration needs before it can name an equivalence bound or a seed
count (docs/preregistration_2026-09-22_draft.md section 5). It trains with mixed-rate sampling -
each batch draws one keep rate - evaluates one checkpoint under every condition, and saves the
prediction for every target frame so the variance can be split three ways. Validation only; the
test split is not read.

    python hdd/d4_pilot.py --features ~/data/hdd/features/dinov2_s1 --split ~/data/hdd/split_a \
        --out ~/data/hdd/d4_pilot
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import metrics
from d3_train import Sessions, drop_history, refuse_inside_git, run_lstm
from models.cfc import CfC


class Temporal(nn.Module):
    """Shared projection and readout; only the recurrent block differs between arms."""

    def __init__(self, kind, dim, hidden=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(dim, hidden)
        self.kind = kind
        self.rnn = nn.LSTM(hidden + 1, hidden, batch_first=True) if kind == "lstm" else CfC(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, f, dt, valid):
        x = self.proj(self.norm(f)) * valid[:, :, None]
        if self.kind == "lstm":
            out, _ = self.rnn(torch.cat([x, dt[:, :, None]], -1))
        else:
            out, _ = self.rnn(x, dt)
        last = valid.float().cumsum(1).argmax(1)
        return self.head(out[torch.arange(len(out), device=out.device), last]).squeeze(-1)

    def block_parameters(self):
        n = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
        return dict(projection=n(self.proj), temporal=n(self.rnn), readout=n(self.head))


def predict(model, data, mu, sd, keep_rate, mask_seed):
    model.eval()
    gen = torch.Generator(device=data.device).manual_seed(mask_seed)
    preds, ys, idx = [], [], []
    with torch.no_grad():
        for f, t, y, b in data.batches(512, shuffle=False):
            mask = (drop_history(t, keep_rate, gen) if keep_rate < 1
                    else torch.ones_like(t, dtype=torch.bool))
            preds.append((run_lstm(model, f, t, mask) * sd + mu).cpu().numpy())
            ys.append(y.cpu().numpy())
    return np.concatenate(preds).astype(np.float32), np.concatenate(ys).astype(np.float64)


def train_once(kind, train, val, lr, seed, epochs, patience, batch, mu, sd, dim, keep_rates):
    """Mixed-rate training: every batch draws one keep rate."""
    torch.manual_seed(seed)
    model = Temporal(kind, dim).to(train.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=train.device).manual_seed(seed)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for f, t, y, _ in train.batches(batch, shuffle=True, rng=rng):
            keep = float(rng.choice(keep_rates))
            mask = (drop_history(t, keep, gen) if keep < 1 else torch.ones_like(t, dtype=torch.bool))
            loss = ((run_lstm(model, f, t, mask) - (y - mu) / sd) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        pred, y = predict(model, val, mu, sd, 1.0, 10_000 + seed)
        score = metrics.macro_mae(pred, y)
        if score < best - 1e-6:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best, ep + 1


def cluster_of(split_dir, sessions):
    info = json.load(open(os.path.join(split_dir, "split.json")))["sessions"]
    return np.array([info[s]["cluster"] for s in sessions])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-hz", type=float, default=10.0)
    ap.add_argument("--window-s", type=float, default=3.0)
    ap.add_argument("--keep-rates", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    ap.add_argument("--main-keep", type=float, default=0.25, help="the pre-registered main condition")
    ap.add_argument("--mask-draws", type=int, default=5)
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-4, 1e-4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
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
    sessions = val.sessions()
    clusters = cluster_of(a.split, sessions)
    print(f"train {len(ytr):,} targets, val {len(yva):,} across {len(set(clusters))} route clusters; "
          f"mixed-rate training over {a.keep_rates}")

    preds = {}
    summary = {}
    for kind in ("lstm", "cfc"):
        runs = []
        for lr in a.lrs:
            _, score, eps = train_once(kind, train, val, lr, a.seeds[0], a.epochs, a.patience,
                                       a.batch, mu, sd, dim, a.keep_rates)
            runs.append(dict(lr=lr, val_macro_mae=score, epochs=eps))
            print(f"  {kind} lr {lr:g}: {score:.3f} ({eps} epochs)")
        best_lr = min(runs, key=lambda r: r["val_macro_mae"])["lr"]
        rows = []
        for seed in a.seeds:
            model, score, eps = train_once(kind, train, val, best_lr, seed, a.epochs, a.patience,
                                           a.batch, mu, sd, dim, a.keep_rates)
            blocks = model.block_parameters()
            for keep in sorted(set(a.keep_rates) | {a.main_keep}):
                draws = a.mask_draws if keep < 1 else 1
                for d in range(draws):
                    p, y = predict(model, val, mu, sd, keep, 1000 * d + seed)
                    preds[f"{kind}_s{seed}_k{keep:g}_m{d}"] = p
                    rows.append(dict(seed=seed, lr=best_lr, epochs=eps, keep=keep, mask=d,
                                     macro_mae=metrics.macro_mae(p, y), ccc=metrics.ccc(p, y)))
            main = [r for r in rows if r["seed"] == seed and r["keep"] == a.main_keep]
            print(f"  {kind} seed {seed} (lr {best_lr:g}, {eps} epochs): full "
                  f"{[r['macro_mae'] for r in rows if r['seed'] == seed and r['keep'] == 1][0]:.3f}, "
                  f"keep {a.main_keep:g} {np.mean([r['macro_mae'] for r in main]):.3f} "
                  f"(masks {np.std([r['macro_mae'] for r in main]):.3f})")
        summary[kind] = dict(lr_search=runs, best_lr=best_lr, rows=rows, parameters=blocks)

    np.savez_compressed(os.path.join(a.out, "val_predictions.npz"), y=yva.astype(np.float32),
                        session=sessions, cluster=clusters, **preds)
    json.dump(dict(base_hz=a.base_hz, window_s=a.window_s, keep_rates=a.keep_rates,
                   main_keep=a.main_keep, mask_draws=a.mask_draws, seeds=a.seeds,
                   summary=summary), open(os.path.join(a.out, "pilot.json"), "w"), indent=1)
    for f in ("val_predictions.npz", "pilot.json"):
        os.chmod(os.path.join(a.out, f), 0o600)
    print(f"saved per-frame predictions for {len(preds)} model/seed/condition/mask combinations "
          f"in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
