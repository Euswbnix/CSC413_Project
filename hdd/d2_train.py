"""D2: can a single frozen frame predict the steering angle? Train, val only, never test.

The gate and its budget were written down before this ran (docs/discussion_2026-09-21_next_steps.md
section 1.3): a fixed head, three learning rates at one seed each, then three seeds at the winner,
at most six runs per encoder, 30 epochs with patience 5. A pass needs all three seeds to reach
macro skill >= 5% against the constant and CCC > 0.3.

Targets are the frames a model is asked about: moving (> 3 m/s) with a steering label that does
not bridge a CAN gap. Metrics come from metrics.py, so they mean what they meant on SullyChen.

    python hdd/d2_train.py --features ~/data/hdd/features/dinov2_s10 --split ~/data/hdd/split_a \
        --out ~/data/hdd/d2/dinov2
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics

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


def load_split(features, split_dir, which):
    """Features, labels, session ids and distance-to-train for every target frame of a split."""
    info = json.load(open(os.path.join(split_dir, "split.json")))
    dist_file = os.path.join(split_dir, "frame_distance.npz")
    dists = np.load(dist_file) if os.path.exists(dist_file) else {}
    X, y, sess, dist = [], [], [], []
    for s, meta in sorted(info["sessions"].items()):
        if meta["split"] != which:
            continue
        p = os.path.join(features, f"{s}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        kept = z["kept"]
        steer, speed = z["steer"][kept], z["speed_mps"][kept]
        m = np.isfinite(steer) & np.isfinite(speed) & (speed > MOVING)
        X.append(z["feats"][m])
        y.append(steer[m])
        sess += [s] * int(m.sum())
        d = dists[s][kept][m] if s in getattr(dists, "files", []) else np.full(int(m.sum()), np.nan, np.float32)
        dist.append(d)
    return (np.concatenate(X), np.concatenate(y).astype(np.float64),
            np.array(sess), np.concatenate(dist), info)


def train_once(Xtr, ytr, Xva, yva, lr, seed, epochs, patience, batch, device):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dim = Xtr.shape[1]
    head = nn.Sequential(nn.LayerNorm(dim, elementwise_affine=False), nn.Linear(dim, 512),
                         nn.GELU(), nn.Linear(512, 1)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    mu, sd = float(ytr.mean()), float(ytr.std())
    xt = torch.from_numpy(Xtr).to(device)
    tt = torch.from_numpy(((ytr - mu) / sd).astype(np.float32)).to(device)
    xv = torch.from_numpy(Xva).to(device)
    best, best_pred, bad = np.inf, None, 0
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(perm), batch):
            j = perm[i:i + batch]
            loss = ((head(xt[j].float()).squeeze(-1) - tt[j]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            pred = np.concatenate([head(xv[i:i + 65536].float()).squeeze(-1).cpu().numpy()
                                   for i in range(0, len(xv), 65536)]) * sd + mu
        score = metrics.macro_mae(pred, yva)
        if score < best - 1e-6:
            best, best_pred, bad = score, pred, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best, best_pred, ep + 1


def report(pred, y, sess, dist, c_val, c_train):
    out = dict(macro_mae=metrics.macro_mae(pred, y), ccc=metrics.ccc(pred, y),
               pred_std=float(np.std(pred)), true_std=float(np.std(y)),
               skill_vs_val_constant=metrics.macro_skill_vs_constant(pred, y, None, c_val),
               skill_vs_train_constant=metrics.macro_skill_vs_constant(pred, y, None, c_train))
    far = dist > 200
    if np.isfinite(dist).any() and far.sum() > 1000:
        out["far_from_train"] = dict(n=int(far.sum()), macro_mae=metrics.macro_mae(pred[far], y[far]),
                                     ccc=metrics.ccc(pred[far], y[far]))
    per = {}
    for s in sorted(set(sess)):
        m = sess == s
        if m.sum() >= 200:
            per[s] = dict(n=int(m.sum()), macro_mae=metrics.macro_mae(pred[m], y[m]),
                          ccc=metrics.ccc(pred[m], y[m]))
    out["per_session"] = per
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-4, 1e-4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4096)
    a = ap.parse_args()
    refuse_inside_git(a.out)
    os.makedirs(a.out, exist_ok=True)
    os.chmod(a.out, 0o700)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr, ytr, str_, dtr, info = load_split(a.features, a.split, "train")
    Xva, yva, sva, dva, _ = load_split(a.features, a.split, "val")
    c_train = metrics.best_constant(ytr)
    c_val = metrics.best_constant(yva)
    print(f"train {len(ytr):,} target frames from {len(set(str_))} sessions; "
          f"val {len(yva):,} from {len(set(sva))} sessions; feature dim {Xtr.shape[1]}")
    print(f"constants: fitted on train {c_train:+.3f} deg (val macro MAE "
          f"{metrics.macro_mae(np.full_like(yva, c_train), yva):.3f}), "
          f"fitted on val {c_val:+.3f} deg ({metrics.macro_mae(np.full_like(yva, c_val), yva):.3f})")

    runs, t0 = [], time.perf_counter()
    for lr in a.lrs:
        score, pred, eps = train_once(Xtr, ytr, Xva, yva, lr, a.seeds[0], a.epochs, a.patience, a.batch, device)
        runs.append(dict(stage="lr_search", lr=lr, seed=a.seeds[0], epochs=eps, val_macro_mae=score))
        print(f"  lr {lr:g}: val macro MAE {score:.3f} after {eps} epochs")
    best_lr = min(runs, key=lambda r: r["val_macro_mae"])["lr"]
    print(f"chosen lr {best_lr:g}")

    finals = []
    for seed in a.seeds:
        score, pred, eps = train_once(Xtr, ytr, Xva, yva, best_lr, seed, a.epochs, a.patience, a.batch, device)
        r = report(pred, yva, sva, dva, c_val, c_train)
        r.update(seed=seed, lr=best_lr, epochs=eps)
        finals.append(r)
        print(f"  seed {seed}: macro MAE {r['macro_mae']:.3f}, CCC {r['ccc']:+.3f}, "
              f"skill vs val constant {r['skill_vs_val_constant']:+.3f}, vs train constant "
              f"{r['skill_vs_train_constant']:+.3f}, pred std {r['pred_std']:.2f} (labels {r['true_std']:.2f})")

    gate = dict(skill_ge_5pct=all(f["skill_vs_val_constant"] >= 0.05 for f in finals),
                ccc_gt_0p3=all(f["ccc"] > 0.3 for f in finals),
                runs_used=len(runs) + len(finals))
    gate["passed"] = gate["skill_ge_5pct"] and gate["ccc_gt_0p3"]
    print(f"D2 gate: skill >= 5% on all seeds {gate['skill_ge_5pct']}, CCC > 0.3 on all seeds "
          f"{gate['ccc_gt_0p3']} -> {'PASS' if gate['passed'] else 'FAIL'} "
          f"({gate['runs_used']} runs, {(time.perf_counter() - t0) / 60:.1f} min)")
    res = dict(features=os.path.basename(a.features), split_parameters=info["parameters"],
               totals=info["totals"], constants=dict(train_fitted=c_train, val_fitted=c_val),
               lr_search=runs, finals=finals, gate=gate)
    p = os.path.join(a.out, "d2_results.json")
    json.dump(res, open(p, "w"), indent=1)
    os.chmod(p, 0o600)


if __name__ == "__main__":
    main()
