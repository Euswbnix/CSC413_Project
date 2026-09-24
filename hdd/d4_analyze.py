"""The pre-registered D4 analysis (docs/preregistration_2026-09-22.md sections 4-5).

Main statistic: macro MAE of arm A minus arm B on the same target frames under a condition,
averaged over seeds and drop masks. Uncertainty: a two-level bootstrap that resamples each arm's
seeds independently (seed i of one arm has nothing to do with seed i of the other) and the route
clusters jointly (the frames are shared). Each cluster's share uses the macro weights of the whole
evaluation set, so the shares add up to the overall macro MAE. Verdict against the equivalence
bound: CI inside +-delta -> practically equivalent; CI excluding 0 -> different; else not
distinguished.

Also reports H-B (degradation from full history to the condition) with the same resampling, and a
per-session and per-date clustering as sensitivity analyses.

    python hdd/d4_analyze.py --pred ~/data/hdd/d4/dinov2 --a cfc --b lstm --split val
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics

DELTA = 0.5


def load(pred_dir, arm, split):
    """{seed: {condition key: per-frame prediction}} plus the shared y, cluster and session."""
    out, ref = {}, None
    for f in sorted(glob.glob(os.path.join(pred_dir, f"pred_{arm}_s*_{split}.npz"))):
        seed = int(re.search(r"_s(\d+)_", os.path.basename(f)).group(1))
        z = np.load(f, allow_pickle=False)
        cur = (z["y"], z["cluster"], z["session"])
        if ref is None:
            ref = cur
        elif not all(np.array_equal(p, q) for p, q in zip(ref, cur)):
            sys.exit(f"{f}: different target frames from the other runs; refusing to pair")
        out[seed] = {k: z[k].astype(np.float64) for k in z.files if k.startswith("k")}
    if not out:
        sys.exit(f"no predictions for {arm} on {split} in {pred_dir}")
    return out, ref


def shares(pred, y, w, groups, keys):
    """Per group, the group's share of macro MAE, averaged over the given mask keys."""
    err = np.mean([w * np.abs(pred[k] - y) for k in keys], axis=0)
    return np.array([err[groups == g].sum() for g in np.unique(groups)])


def keys_for(runs, keep):
    tag = f"k{keep:g}_m"
    return sorted(k for k in next(iter(runs.values())) if k.startswith(tag))


def bootstrap(sa, sb, n_boot, rng):
    """sa, sb: (seeds, groups) share matrices. Returns the resampled differences of the mean total."""
    na, nb, g = sa.shape[0], sb.shape[0], sa.shape[1]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        cg = rng.integers(g, size=g)
        diffs[i] = sa[rng.integers(na, size=na)][:, cg].sum(1).mean() - \
                   sb[rng.integers(nb, size=nb)][:, cg].sum(1).mean()
    return diffs


def verdict(lo, hi, delta):
    if -delta < lo and hi < delta:
        return "practically equivalent"
    if lo > 0 or hi < 0:
        return "different"
    return "not distinguished"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--a", default="cfc")
    ap.add_argument("--b", default="lstm")
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--main-keep", type=float, default=0.25)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out")
    a = ap.parse_args()
    ra, (y, cluster, session) = load(a.pred, a.a, a.split)
    rb, (yb, cb, sb_) = load(a.pred, a.b, a.split)
    if not (np.array_equal(y, yb) and np.array_equal(cluster, cb)):
        sys.exit("the two arms were scored on different frames")
    w = metrics.macro_weights(y)
    date = np.array([s[:8] for s in session])
    rng = np.random.default_rng(0)
    res = dict(a=a.a, b=a.b, split=a.split, seeds_a=sorted(ra), seeds_b=sorted(rb), delta=a.delta,
               frames=int(len(y)), clusters=int(len(np.unique(cluster))), conditions={})
    print(f"{a.a} ({len(ra)} seeds) minus {a.b} ({len(rb)} seeds) on {a.split}: {len(y):,} frames, "
          f"{res['clusters']} route clusters; delta {a.delta}")
    keeps = sorted({float(k.split('_')[0][1:]) for k in next(iter(ra.values()))}, reverse=True)
    mats = {}
    for keep in keeps:
        ka, kb = keys_for(ra, keep), keys_for(rb, keep)
        sa = np.array([shares(ra[s], y, w, cluster, ka) for s in sorted(ra)])
        sbm = np.array([shares(rb[s], y, w, cluster, kb) for s in sorted(rb)])
        mats[keep] = (sa, sbm)
        est = sa.sum(1).mean() - sbm.sum(1).mean()
        d = bootstrap(sa, sbm, a.boot, rng)
        lo, hi = np.percentile(d, [2.5, 97.5])
        row = dict(estimate=float(est), ci=[float(lo), float(hi)], verdict=verdict(lo, hi, a.delta),
                   mae_a=float(sa.sum(1).mean()), mae_b=float(sbm.sum(1).mean()),
                   seed_sd_a=float(sa.sum(1).std(ddof=1)), seed_sd_b=float(sbm.sum(1).std(ddof=1)))
        # sensitivity: the same resampling with sessions, then dates, as the clusters
        for name, groups in (("by_session", session), ("by_date", date)):
            ga = np.array([shares(ra[s], y, w, groups, ka) for s in sorted(ra)])
            gb = np.array([shares(rb[s], y, w, groups, kb) for s in sorted(rb)])
            lo2, hi2 = np.percentile(bootstrap(ga, gb, a.boot, rng), [2.5, 97.5])
            row[name] = dict(ci=[float(lo2), float(hi2)], verdict=verdict(lo2, hi2, a.delta),
                             groups=int(len(np.unique(groups))))
        res["conditions"][f"keep_{keep:g}"] = row
        mark = "  <- main" if keep == a.main_keep else ""
        print(f"  keep {keep:4g}: {a.a} {row['mae_a']:.3f}  {a.b} {row['mae_b']:.3f}  diff {est:+.3f}  "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]  {row['verdict']}{mark}")
        print(f"             sensitivity: by session {row['by_session']['verdict']} "
              f"[{row['by_session']['ci'][0]:+.3f}, {row['by_session']['ci'][1]:+.3f}], "
              f"by date {row['by_date']['verdict']} "
              f"[{row['by_date']['ci'][0]:+.3f}, {row['by_date']['ci'][1]:+.3f}]")
    # H-B: degradation from full history to the main condition, A minus B
    if 1.0 in mats and a.main_keep in mats:
        da = mats[a.main_keep][0] - mats[1.0][0]
        db = mats[a.main_keep][1] - mats[1.0][1]
        est = da.sum(1).mean() - db.sum(1).mean()
        lo, hi = np.percentile(bootstrap(da, db, a.boot, rng), [2.5, 97.5])
        res["degradation"] = dict(a=float(da.sum(1).mean()), b=float(db.sum(1).mean()),
                                  estimate=float(est), ci=[float(lo), float(hi)])
        print(f"  H-B degradation full -> keep {a.main_keep:g}: {a.a} {da.sum(1).mean():+.3f}, "
              f"{a.b} {db.sum(1).mean():+.3f}, difference {est:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        os.chmod(a.out, 0o600)


if __name__ == "__main__":
    main()
