"""Is the LTC's ODE integration converged at 6 sub-steps? Evaluation only, fixed trained weights.

The pre-registration (section 6) fixes the rule before this runs: seed 0's weights, the 75% drop
condition, five masks each; if 6 and 24 sub-steps differ by more than 0.1 degrees of macro MAE,
every LTC run is retrained at 24. The number of sub-steps changes only how finely the cell
integrates between observations, not its parameters, so the same state dict loads into each.

    python hdd/ltc_unfolds_check.py --cache ~/data/hdd/cache/dinov2_10hz \
        --ckpt ~/data/hdd/d4/dinov2/ckpt/ltc_lr0.001_s0.pt --out ~/data/hdd/d4/dinov2/ltc_unfolds_check.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import d4_run as D
import metrics

THRESHOLD = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--unfolds", type=int, nargs="+", default=[6, 12, 24])
    ap.add_argument("--keep", type=float, default=0.25)
    ap.add_argument("--masks", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    D.refuse_inside_git(a.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train = D.Cached(a.cache, "train", 30, dev)
    val = D.Cached(a.cache, "val", 30, dev)
    ytr, yva = train.labels(), val.labels()
    mu, sd = float(ytr.mean()), float(ytr.std())
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)["best_state"]
    rows = {}
    for k in a.unfolds:
        model = D.Arm("ltc", train.items[0]["feats"].shape[1], seed=a.seed, ode_unfolds=k).to(dev)
        model.load_state_dict(state)
        scores = [metrics.macro_mae(D.predict(model, val, mu, sd, a.keep, 1000 * m + 7), yva)
                  for m in range(a.masks)]
        full = metrics.macro_mae(D.predict(model, val, mu, sd, 1.0, 7), yva)
        rows[k] = dict(keep=float(np.mean(scores)), keep_masks=scores, full=full)
        print(f"ode_unfolds {k:2d}: keep {a.keep:g} macro MAE {np.mean(scores):.4f} "
              f"(masks sd {np.std(scores):.4f}), full history {full:.4f}", flush=True)
    gap = abs(rows[max(a.unfolds)]["keep"] - rows[min(a.unfolds)]["keep"])
    verdict = "retrain at %d" % max(a.unfolds) if gap > THRESHOLD else "keep %d" % min(a.unfolds)
    print(f"|{min(a.unfolds)} vs {max(a.unfolds)}| = {gap:.4f} deg, threshold {THRESHOLD} -> {verdict}")
    json.dump(dict(rows={str(k): v for k, v in rows.items()}, gap=gap, threshold=THRESHOLD,
                   verdict=verdict, ckpt=os.path.basename(a.ckpt)), open(a.out, "w"), indent=1)
    os.chmod(a.out, 0o600)


if __name__ == "__main__":
    main()
