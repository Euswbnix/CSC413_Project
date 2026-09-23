"""Split the CfC-minus-LSTM difference into seed, route-cluster and drop-mask variance.

Reads the pilot's per-frame validation predictions and answers what the pre-registration needs:
how large is the paired difference, how much does it move with the training seed, with the route
cluster, and with the random drop mask, and therefore what equivalence bound and seed count are
worth committing to.

The per-cluster contribution uses the macro weights of the whole evaluation set, so the
contributions add up to the overall difference (metrics.macro_weights).

    python hdd/d4_variance.py --pilot ~/data/hdd/d4_pilot/dinov2
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics


def contributions(pred_a, pred_b, y, w, cluster):
    """Per-cluster share of the macro-MAE difference (A minus B); the sum is the total."""
    d = w * (np.abs(pred_a - y) - np.abs(pred_b - y))
    out = {}
    for c in np.unique(cluster):
        out[int(c)] = float(d[cluster == c].sum())
    return out, float(d.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--a", default="cfc")
    ap.add_argument("--b", default="lstm")
    ap.add_argument("--keep", type=float, default=0.25)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.pilot, "pilot.json")))
    z = np.load(os.path.join(a.pilot, "val_predictions.npz"))
    y = z["y"].astype(np.float64)
    cluster = z["cluster"]
    w = metrics.macro_weights(y)
    seeds, draws = cfg["seeds"], cfg["mask_draws"]

    rows, totals = [], {}
    for s in seeds:
        for m in range(draws):
            ka = f"{a.a}_s{s}_k{a.keep:g}_m{m}"
            kb = f"{a.b}_s{s}_k{a.keep:g}_m{m}"
            if ka not in z.files or kb not in z.files:
                continue
            per, tot = contributions(z[ka].astype(np.float64), z[kb].astype(np.float64), y, w, cluster)
            totals[(s, m)] = tot
            rows.append(dict(seed=s, mask=m, total=tot, per_cluster=per))
    if not rows:
        sys.exit("no matching predictions; check --a/--b/--keep")

    tot = np.array([r["total"] for r in rows])
    by_seed = {s: np.array([r["total"] for r in rows if r["seed"] == s]) for s in seeds}
    seed_means = np.array([v.mean() for v in by_seed.values()])
    mask_dev = np.concatenate([v - v.mean() for v in by_seed.values()])
    clusters = sorted(rows[0]["per_cluster"])
    per_cluster = np.array([[r["per_cluster"][c] for c in clusters] for r in rows])

    print(f"paired difference {a.a} minus {a.b} at keep {a.keep:g}, in macro MAE degrees")
    print(f"  overall mean {tot.mean():+.3f}  (negative means {a.a} is better)")
    print(f"  across seeds  : mean {seed_means.mean():+.3f}, sd {seed_means.std(ddof=1):.3f}, "
          f"values {np.round(seed_means, 3).tolist()}")
    print(f"  across masks  : sd within a seed {mask_dev.std(ddof=1):.4f}")
    print(f"  clusters      : {len(clusters)}; contribution sd across clusters "
          f"{per_cluster.mean(0).std(ddof=1):.3f}, largest |contribution| "
          f"{np.abs(per_cluster.mean(0)).max():.3f}")

    # what the test set would give: resample clusters (seed x cluster, two levels)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        s = rng.choice(seeds, len(seeds))
        c = rng.choice(len(clusters), len(clusters))
        vals = []
        for si in s:
            r = [row for row in rows if row["seed"] == si]
            pick = r[rng.integers(len(r))]                 # one mask draw for this seed
            vals.append(sum(pick["per_cluster"][clusters[j]] for j in c))
        boot.append(np.mean(vals))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    half = (hi - lo) / 2
    print(f"  seed x cluster bootstrap: 95% CI [{lo:+.3f}, {hi:+.3f}], half-width {half:.3f} deg")
    print(f"  with n seeds the half-width scales roughly as sqrt(3/n) on the seed part:")
    for n in (3, 5, 8, 10):
        print(f"    n={n:2d}: approx half-width {half * np.sqrt(3 / n):.3f} deg")
    json.dump(dict(keep=a.keep, a=a.a, b=a.b, rows=rows,
                   seed_sd=float(seed_means.std(ddof=1)), mask_sd=float(mask_dev.std(ddof=1)),
                   ci=[float(lo), float(hi)]), open(os.path.join(a.pilot, "variance.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
