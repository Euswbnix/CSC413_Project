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

--family runs the whole registered comparison set (section 5): the one primary test, CfC minus
LSTM+dt at 75% dropped, judged by its 95% interval exactly as above; and the secondary family --
every other condition of CfC vs LSTM+dt, LTC vs LSTM+dt and LTC vs CfC at every condition, and H-B
for each pair -- Holm-corrected together. Each secondary comparison gets two bootstrap p-values:
p_diff for "the difference is 0" and p_equiv for "the difference lies outside +-delta" (at 0.05
they reproduce "the 95% interval excludes 0" / "lies inside +-delta"). Holm runs over the family
separately for the two kinds of claim; within one comparison the two nulls cannot both hold. H-B
has no registered equivalence bound, so it gets only p_diff. The Transformer (section 9.1) is
descriptive and never enters either family.

    python hdd/d4_analyze.py --pred ~/workspace/hdd/d4/dinov2 --a cfc --b lstm --split val
    python hdd/d4_analyze.py --pred ~/workspace/hdd/d4/dinov2 --family --split val --out family_val.json
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
ALPHA = 0.05
REGISTERED_PAIRS = (("cfc", "lstm"), ("ltc", "lstm"), ("ltc", "cfc"))
PRIMARY = ("cfc", "lstm", 0.25)


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


def p_values(d, delta):
    """Bootstrap p-values from the resampled differences d: two-sided for 'difference is 0', and
    TOST for 'difference is outside +-delta'. (count + 1) / (B + 1) keeps them away from 0."""
    B = len(d)
    le = lambda x: (np.sum(d <= x) + 1) / (B + 1)
    ge = lambda x: (np.sum(d >= x) + 1) / (B + 1)
    return (float(min(1.0, 2 * min(le(0.0), ge(0.0)))),
            float(min(1.0, 2 * max(ge(delta), le(-delta)))) if delta is not None else None)


def holm(p):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(p, dtype=float)
    adj, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


def analyze(pred, arm_a, arm_b, split, main_keep=0.25, delta=DELTA, n_boot=10000, show=True):
    say = print if show else (lambda *x, **k: None)
    ra, (y, cluster, session) = load(pred, arm_a, split)
    rb, (yb, cb, sb_) = load(pred, arm_b, split)
    if not (np.array_equal(y, yb) and np.array_equal(cluster, cb)):
        sys.exit("the two arms were scored on different frames")
    w = metrics.macro_weights(y)
    date = np.array([s[:8] for s in session])
    rng = np.random.default_rng(0)
    res = dict(a=arm_a, b=arm_b, split=split, seeds_a=sorted(ra), seeds_b=sorted(rb), delta=delta,
               frames=int(len(y)), clusters=int(len(np.unique(cluster))), conditions={})
    say(f"{arm_a} ({len(ra)} seeds) minus {arm_b} ({len(rb)} seeds) on {split}: {len(y):,} frames, "
          f"{res['clusters']} route clusters; delta {delta}")
    keeps = sorted({float(k.split('_')[0][1:]) for k in next(iter(ra.values()))}, reverse=True)
    mats = {}
    for keep in keeps:
        ka, kb = keys_for(ra, keep), keys_for(rb, keep)
        sa = np.array([shares(ra[s], y, w, cluster, ka) for s in sorted(ra)])
        sbm = np.array([shares(rb[s], y, w, cluster, kb) for s in sorted(rb)])
        mats[keep] = (sa, sbm)
        est = sa.sum(1).mean() - sbm.sum(1).mean()
        d = bootstrap(sa, sbm, n_boot, rng)
        lo, hi = np.percentile(d, [2.5, 97.5])
        p_diff, p_equiv = p_values(d, delta)
        row = dict(estimate=float(est), ci=[float(lo), float(hi)], verdict=verdict(lo, hi, delta),
                   p_diff=p_diff, p_equiv=p_equiv,
                   mae_a=float(sa.sum(1).mean()), mae_b=float(sbm.sum(1).mean()),
                   seed_sd_a=float(sa.sum(1).std(ddof=1)), seed_sd_b=float(sbm.sum(1).std(ddof=1)))
        # sensitivity: the same resampling with sessions, then dates, as the clusters
        for name, groups in (("by_session", session), ("by_date", date)):
            ga = np.array([shares(ra[s], y, w, groups, ka) for s in sorted(ra)])
            gb = np.array([shares(rb[s], y, w, groups, kb) for s in sorted(rb)])
            lo2, hi2 = np.percentile(bootstrap(ga, gb, n_boot, rng), [2.5, 97.5])
            row[name] = dict(ci=[float(lo2), float(hi2)], verdict=verdict(lo2, hi2, delta),
                             groups=int(len(np.unique(groups))))
        res["conditions"][f"keep_{keep:g}"] = row
        mark = "  <- main" if keep == main_keep else ""
        say(f"  keep {keep:4g}: {arm_a} {row['mae_a']:.3f}  {arm_b} {row['mae_b']:.3f}  diff {est:+.3f}  "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]  {row['verdict']}{mark}")
        say(f"             sensitivity: by session {row['by_session']['verdict']} "
              f"[{row['by_session']['ci'][0]:+.3f}, {row['by_session']['ci'][1]:+.3f}], "
              f"by date {row['by_date']['verdict']} "
              f"[{row['by_date']['ci'][0]:+.3f}, {row['by_date']['ci'][1]:+.3f}]")
    # H-B: degradation from full history to the main condition, A minus B
    if 1.0 in mats and main_keep in mats:
        da = mats[main_keep][0] - mats[1.0][0]
        db = mats[main_keep][1] - mats[1.0][1]
        est = da.sum(1).mean() - db.sum(1).mean()
        dd = bootstrap(da, db, n_boot, rng)
        lo, hi = np.percentile(dd, [2.5, 97.5])
        res["degradation"] = dict(a=float(da.sum(1).mean()), b=float(db.sum(1).mean()),
                                  estimate=float(est), ci=[float(lo), float(hi)],
                                  p_diff=p_values(dd, None)[0])
        say(f"  H-B degradation full -> keep {main_keep:g}: {arm_a} {da.sum(1).mean():+.3f}, "
              f"{arm_b} {db.sum(1).mean():+.3f}, difference {est:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return res


def family(pred, split, main_keep=0.25, delta=DELTA, n_boot=10000):
    """The primary test plus the Holm-corrected secondary family over whatever registered pairs
    have predictions (a pair that was not evaluated is listed as missing, not silently dropped)."""
    pairs, missing = {}, []
    for arm_a, arm_b in REGISTERED_PAIRS:
        if all(glob.glob(os.path.join(pred, f"pred_{arm}_s*_{split}.npz")) for arm in (arm_a, arm_b)):
            print(f"\n== {arm_a} vs {arm_b}")
            pairs[f"{arm_a}-{arm_b}"] = analyze(pred, arm_a, arm_b, split, main_keep, delta, n_boot)
        else:
            missing.append(f"{arm_a}-{arm_b}")
    primary, tests = None, []
    for key, r in pairs.items():
        arm_a, arm_b = key.split("-")
        for cond, row in r["conditions"].items():
            item = dict(test=f"{key} {cond}", estimate=row["estimate"], ci=row["ci"],
                        p_diff=row["p_diff"], p_equiv=row["p_equiv"])
            if (arm_a, arm_b, float(cond.split("_", 1)[1])) == PRIMARY:
                primary = dict(item, verdict=row["verdict"])
            else:
                tests.append(item)
        if "degradation" in r:
            g = r["degradation"]
            tests.append(dict(test=f"{key} H-B", estimate=g["estimate"], ci=g["ci"],
                              p_diff=g["p_diff"], p_equiv=None))
    # the smallest bootstrap p-value is 2 / (B + 1); Holm multiplies it by up to m, so too few draws
    # would make every secondary test unreachable however large the effect
    if tests and 2 * len(tests) / (n_boot + 1) > ALPHA / 5:
        sys.exit(f"{n_boot} bootstrap draws cannot resolve Holm over {len(tests)} tests; use more")
    adj_diff = holm([t["p_diff"] for t in tests])
    eq_idx = [i for i, t in enumerate(tests) if t["p_equiv"] is not None]
    adj_eq = dict(zip(eq_idx, holm([tests[i]["p_equiv"] for i in eq_idx])))
    for i, t in enumerate(tests):
        t["p_diff_holm"] = float(adj_diff[i])
        t["p_equiv_holm"] = float(adj_eq[i]) if i in adj_eq else None
        t["verdict_holm"] = ("practically equivalent" if i in adj_eq and adj_eq[i] <= ALPHA else
                             "different" if adj_diff[i] <= ALPHA else "not distinguished")
    print(f"\n== {split}: primary test (no correction) and secondary family (Holm, m = {len(tests)}, "
          f"alpha {ALPHA})" + (f"; not evaluated: {', '.join(missing)}" if missing else ""))
    if primary:
        print(f"  PRIMARY {primary['test']:24s} {primary['estimate']:+.3f} [{primary['ci'][0]:+.3f}, "
              f"{primary['ci'][1]:+.3f}]  {primary['verdict']}")
    for t in tests:
        eq = f"{t['p_equiv_holm']:.4f}" if t["p_equiv_holm"] is not None else "   -  "
        print(f"  {t['test']:32s} {t['estimate']:+.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}]  "
              f"p_diff {t['p_diff']:.4f} -> Holm {t['p_diff_holm']:.4f}  p_equiv Holm {eq}  {t['verdict_holm']}")
    return dict(split=split, alpha=ALPHA, delta=delta, primary=primary, secondary=tests,
                secondary_m=len(tests), missing_pairs=missing, pairs=pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--a", default="cfc")
    ap.add_argument("--b", default="lstm")
    ap.add_argument("--family", action="store_true", help="all registered pairs, Holm on the secondary family")
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--main-keep", type=float, default=0.25)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = (family(a.pred, a.split, a.main_keep, a.delta, a.boot) if a.family else
           analyze(a.pred, a.a, a.b, a.split, a.main_keep, a.delta, a.boot))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        os.chmod(a.out, 0o600)




if __name__ == "__main__":
    main()
