#!/usr/bin/env python3
"""Add the new metrics to every evaluated run from its saved predictions -- no inference.

For each run with both `predictions_{split}.npz` and `final_metrics_{split}.json`, rebuild
the rollout summary with `metrics.summary(..., ref_constant=...)` and merge the NEW keys into
the stored block. The old keys are recomputed too and must agree with what was reported; if
they do not, the predictions file does not belong to the reported numbers and the run is
skipped rather than silently overwritten. Pearson r is compared NaN-aware, because r of an
exactly constant prediction is now NaN by construction (see metrics.pearson_r).

    python scripts/recompute_metrics.py lab runs_power runs_reg_* --processed data/processed
"""
import argparse, json, math, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import metrics

OLD = ("macro_skill", "macro_mae", "macro_mae_predict0", "global_mae", "pearson_r",
       "constant_prediction", "false_alarm_rate", "n_valid")
NEW = ("ccc", "oracle_constant", "macro_mae_oracle_constant", "ref_constant",
       "macro_mae_ref_constant", "macro_skill_vs_ref_constant")


def same(a, b, tol=1e-9):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if a is None or b is None:
        return a is b
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=pathlib.Path)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    c_ref = metrics.reference_constant(a.processed, "val")
    print(f"reference constant (best on val) = {c_ref:+.4f} deg")
    done = skipped = mismatched = 0
    for root in a.roots:
        for npz in sorted(root.rglob("predictions_*.npz")):
            split = npz.stem.split("_", 1)[1]
            js = npz.parent / f"final_metrics_{split}.json"
            if not js.exists():
                skipped += 1
                continue
            z = np.load(npz)
            p, y, v = z["pred"], z["true"], z["valid"].astype(bool)
            d = json.loads(js.read_text())
            old = d.get("rollout") or {}
            s = metrics.summary(p, y, v, ref_constant=c_ref)
            # global_mae was accumulated in float32 when first reported, which is only good to
            # ~1e-7 and platform-dependent; everything else was float64 and must match exactly.
            tol = lambda k: 1e-5 if k == "global_mae" else 1e-9
            bad = [k for k in OLD if k in old and not same(old[k], s[k], tol(k))]
            if bad:
                mismatched += 1
                print(f"  MISMATCH {npz.parent.name} [{split}]: " +
                      ", ".join(f"{k} {old[k]} -> {s[k]}" for k in bad))
                continue
            for k in NEW:
                old[k] = s[k]
            d["rollout"] = old
            d["metrics_version"] = 2
            if not a.dry_run:
                js.write_text(json.dumps(d, indent=2) + "\n")
            done += 1
    print(f"updated {done}, skipped (no json) {skipped}, mismatched (left untouched) {mismatched}")
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
