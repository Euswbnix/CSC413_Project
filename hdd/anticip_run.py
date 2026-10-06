"""Steering anticipation: from frames up to now, predict the steering angle 0, 0.5, 1 and 2 s ahead.

Everything here is fixed by docs/plan_2026-10-05_steering_anticipation.md, written before any run.
The data, split, window rule, arms, optimiser, budget and mixed-rate training are D4's (the
training loop is d4_run.train itself); what changes is the target and the read-out:

  * anchors: D4's target frames, kept only where the label exists at all four horizons, so every
    horizon is scored on the same frames;
  * label at t + h: the cached 10 Hz steering series interpolated linearly in camera time, void if
    the two frames around t + h are more than 0.2 s apart, not finite, or out of order;
  * one head with four outputs, each standardised with its own train mean and sd;
  * selection: mean macro MAE over the three future horizons and the three drop conditions;
  * a fifth arm, "frame", sees only the current frame (no history).

    python hdd/anticip_run.py --cache cache/dinov2_10hz --arm lstm --stage lr --out d4/anticip
    python hdd/anticip_run.py --cache cache/dinov2_10hz --arm lstm --stage final --lr 3e-4 --out d4/anticip
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import d4_run as D
import metrics
import tracking

HORIZONS = (0.0, 0.5, 1.0, 2.0)          # seconds ahead of the last observed frame
FUTURE = (1, 2, 3)                        # the columns selection is based on
ARMS = ("frame", "lstm", "cfc", "ltc", "transformer")
PROJECT = "CSC413-anticipation"
MAX_GAP = 0.2                             # as D4's window rule
LOOKAHEAD = 64                            # frames searched for t + h (2 s is 20 at 10 Hz)


def future_labels(t, steer, anchors, horizons=HORIZONS, max_gap=MAX_GAP):
    """(len(anchors), len(horizons)) steering at t[anchors] + h, NaN where it cannot be trusted:
    the two frames around t + h are more than `max_gap` apart, either is not finite, the clock
    steps backwards between the anchor and them, or the session ends first."""
    t, steer, anchors = np.asarray(t, dtype=np.float64), np.asarray(steer, dtype=np.float64), np.asarray(anchors)
    out = np.full((len(anchors), len(horizons)), np.nan)
    back = np.concatenate([[0], np.cumsum(np.diff(t) <= 0)])          # backward steps so far
    ahead = np.minimum(anchors[:, None] + np.arange(LOOKAHEAD)[None], len(t) - 1)
    for k, h in enumerate(horizons):
        if h == 0:
            out[:, k] = steer[anchors]
            continue
        tau = t[anchors] + h
        reached = t[ahead] >= tau[:, None]
        j1 = ahead[np.arange(len(anchors)), reached.argmax(1)]        # first frame at or after t + h
        j0 = j1 - 1
        gap = t[j1] - t[j0]
        ok = (reached.any(1) & (j1 > anchors) & (gap > 0) & (gap <= max_gap)
              & np.isfinite(steer[j0]) & np.isfinite(steer[j1]) & (back[j1] == back[anchors]))
        w = np.divide(tau - t[j0], gap, out=np.zeros_like(gap), where=gap > 0)
        out[ok, k] = (steer[j0] * (1 - w) + steer[j1] * w)[ok]
    return out


class AnticipCached(D.Cached):
    """D4's windows, with the four future labels per anchor and only the anchors that have all."""

    def __init__(self, cache, which, steps, device, max_step=0.2):
        super().__init__(cache, which, steps, device, max_step)
        kept = []
        for it in self.items:
            Y = future_labels(it["t"], it["y"], it["tgt"])
            ok = np.isfinite(Y).all(1)
            if ok.any():
                it["tgt"], it["Y"] = it["tgt"][ok], Y[ok].astype(np.float32)
                kept.append(it)
        self.items = kept
        self.n = int(sum(len(it["tgt"]) for it in kept))

    def labels(self):
        return np.concatenate([it["Y"] for it in self.items]).astype(np.float64)

    def batches(self, batch, shuffle, rng=None):
        order = np.arange(len(self.items))
        if shuffle:
            rng.shuffle(order)
        for i in order:
            it = self.items[i]
            idx = rng.permutation(len(it["tgt"])) if shuffle else np.arange(len(it["tgt"]))
            for j in range(0, len(idx), batch):
                sel = idx[j:j + batch]
                win = it["tgt"][sel][:, None] - np.arange(self.steps - 1, -1, -1)[None]
                f = torch.from_numpy(np.ascontiguousarray(it["feats"][win])).to(self.device).float()
                t = torch.from_numpy(it["t"][win]).to(self.device)
                yield f, t, torch.from_numpy(it["Y"][sel]).to(self.device)


def score_all(model, data, y, mu, sd, keeps, draws):
    """Selection score (mean macro MAE over the future horizons and the conditions), the
    per-condition, per-horizon numbers, and the (N, 4) predictions."""
    per, preds = {}, {}
    for keep in keeps:
        rows = []
        for d in range(draws if keep < 1 else 1):
            p = D.predict(model, data, mu, sd, keep, 1000 * d + 7)
            preds[f"k{keep:g}_m{d}"] = p
            rows.append(dict(mask=d,
                             macro_mae=[metrics.macro_mae(p[:, k], y[:, k]) for k in range(len(HORIZONS))],
                             ccc=[metrics.ccc(p[:, k], y[:, k]) for k in range(len(HORIZONS))]))
        per[f"keep_{keep:g}"] = rows
    # as in D4: average within a condition first (its mask draws, here also the future horizons),
    # then over the conditions, so the full-history condition is not outvoted by its single draw
    mean = float(np.mean([np.mean([[r["macro_mae"][k] for k in FUTURE] for r in rows])
                          for rows in per.values()]))
    return mean, per, preds


def log_metrics(per, prefix):
    """{prefix}/h1.0s/keep25/macro_mae ... (mean over mask draws)."""
    out = {}
    for key, rows in per.items():
        pct = round(float(key.split("_", 1)[1]) * 100)
        for k, h in enumerate(HORIZONS):
            out[f"{prefix}/h{h:g}s/keep{pct}/macro_mae"] = np.mean([r["macro_mae"][k] for r in rows])
            out[f"{prefix}/h{h:g}s/keep{pct}/ccc"] = np.mean([r["ccc"][k] for r in rows])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--stage", required=True, choices=("lr", "final"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-name", default="val", choices=("val", "test"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-4, 1e-4, 3e-5])
    ap.add_argument("--lr", type=float, help="final stage: the rate chosen by the lr stage")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--keep-rates", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    ap.add_argument("--mask-draws", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ode-unfolds", type=int, default=24)
    ap.add_argument("--compile-ltc", action="store_true")
    ap.add_argument("--max-step", type=float, default=0.2)
    ap.add_argument("--tag", default="", help="suffix for the output files, so parallel runs do not collide")
    tracking.add_argument(ap)
    a = ap.parse_args()
    a.dt_unit, a.time_code, a.project, a.horizons = 1.0, "real", PROJECT, list(HORIZONS)
    D.refuse_inside_git(a.out)
    os.makedirs(a.out, exist_ok=True)
    os.chmod(a.out, 0o700)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    train_d = AnticipCached(a.cache, "train", a.steps, device, a.max_step)
    val_d = AnticipCached(a.cache, a.split_name, a.steps, device, a.max_step)
    ytr, yva = train_d.labels(), val_d.labels()
    mu = torch.tensor(ytr.mean(0), dtype=torch.float32, device=device)
    sd = torch.tensor(ytr.std(0), dtype=torch.float32, device=device)
    dim = train_d.items[0]["feats"].shape[1]
    tag = f"{a.arm}_{a.stage}" + (f"_s{'-'.join(map(str, a.seeds))}" if a.stage == "final" else "")
    print(f"[{tag}] train {len(ytr):,} anchors, {a.split_name} {len(yva):,}; horizons {list(HORIZONS)} s; "
          f"dim {dim}; device {device}")
    kw = dict(score=score_all, to_log=log_metrics, n_out=len(HORIZONS))

    rows = []
    if a.stage == "lr":
        for lr in a.lrs:
            _, mean, eps, div, run = D.train(a.arm, train_d, val_d, yva, lr, a.seeds[0], a, mu, sd, dim, tag, **kw)
            run.finish()
            rows.append(dict(lr=lr, seed=a.seeds[0], mean_macro_mae=mean, epochs=eps, diverged=div,
                             ode_unfolds=a.ode_unfolds if a.arm == "ltc" else None))
            print(f"[{tag}] lr {lr:g}: mean over future horizons and conditions {mean:.3f} ({eps} epochs)")
        best = min(rows, key=lambda r: r["mean_macro_mae"])["lr"]
        print(f"[{tag}] chosen lr {best:g}")
        json.dump(dict(arm=a.arm, stage="lr", rows=rows, chosen_lr=best),
                  open(os.path.join(a.out, f"{a.arm}_lr{a.tag}.json"), "w"), indent=1)
    else:
        if not a.lr:
            sys.exit("--lr is required for the final stage")
        for seed in a.seeds:
            done = os.path.join(a.out, f"{a.arm}_s{seed}_{a.split_name}.json")
            if os.path.exists(done):
                print(f"[{tag}] seed {seed}: already finished and evaluated ({os.path.basename(done)})")
                continue
            model, mean, eps, div, run = D.train(a.arm, train_d, val_d, yva, a.lr, seed, a, mu, sd, dim, tag, **kw)
            final_mean, per, preds = score_all(model, val_d, yva, mu, sd, a.keep_rates, a.mask_draws)
            run.log({f"final_{a.split_name}/mean_macro_mae": final_mean,
                     **log_metrics(per, f"final_{a.split_name}")}, step=eps)
            run.finish()
            np.savez_compressed(os.path.join(a.out, f"pred_{a.arm}_s{seed}_{a.split_name}.npz"),
                                y=yva.astype(np.float32), cluster=val_d.clusters(),
                                session=val_d.sessions(), horizons=np.array(HORIZONS), **preds)
            row = dict(arm=a.arm, seed=seed, lr=a.lr, epochs=eps, mean_macro_mae=mean, diverged=div,
                       horizons=list(HORIZONS), ode_unfolds=a.ode_unfolds if a.arm == "ltc" else None,
                       blocks=model.blocks(), conditions=per)
            json.dump(row, open(done, "w"), indent=1)
            rows.append(row)
            full = per["keep_1"][0]["macro_mae"]
            print(f"[{tag}] seed {seed} ({eps} epochs), full history: "
                  + " ".join(f"{h:g}s {m:.3f}" for h, m in zip(HORIZONS, full)))
    for f in os.listdir(a.out):
        if os.path.isfile(os.path.join(a.out, f)):
            os.chmod(os.path.join(a.out, f), 0o600)
    print(f"[{tag}] done in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
