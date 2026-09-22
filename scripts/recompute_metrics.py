#!/usr/bin/env python3
"""Add the new metrics to every evaluated run from its saved predictions -- no inference.

For each run with both `predictions_{split}.npz` and `final_metrics_{split}.json`, rebuild
the rollout summary with `metrics.summary(..., ref_constant=...)` and merge the NEW keys into
the stored block. The old keys are recomputed too and must agree with what was reported; if
they do not, the predictions file does not belong to the reported numbers and the run is
skipped rather than silently overwritten. Pearson r is compared NaN-aware, because r of an
exactly constant prediction is now NaN by construction (see metrics.pearson_r).

    python scripts/recompute_metrics.py lab runs_power runs_reg_* --processed data/processed

It also reports how many distinct label/validity arrays each split has across the runs (1 means
every run was scored against the same frames) and a summary per CONFIGURATION: the run name with
the seed removed, plus the config keys that are not in the name (epochs, max_steps,
per_frame_bug, processed). Runs of one configuration are pooled across the roots given, one per
seed (the first root listed wins), so e.g. `runs_val/runs_fleet runs_power` gives the 35-seed
main comparison. Per configuration and split it prints the mean macro MAE, per-bin MAE, how many
runs are near-constant (prediction std below 1% of the label std), and, from metrics.csv, the
median first -> last training loss and validation MSE. Use --dry-run to only check and summarise;
without it the new metric keys are written back into final_metrics_*.json.
"""
import argparse, csv, json, math, pathlib, re, sys
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


NOT_IN_NAME = ("epochs", "max_steps", "per_frame_bug", "processed")


def config_key(run):
    """Seed-free run name plus the config values the name does not encode."""
    cfg_path = run / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    extra = tuple(str(pathlib.Path(str(cfg[k])).name) if k == "processed" else str(cfg[k])
                  for k in NOT_IN_NAME if cfg.get(k) is not None)
    return (re.sub(r"_s\d+_", "_", run.name),) + extra, cfg.get("seed", run.name)


def first_last(run):
    """(train_loss first, last, val MSE first, last) from metrics.csv, or None."""
    f = run / "metrics.csv"
    if not f.exists():
        return None
    with f.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "train_loss" not in rows[0]:
        return None
    col = "val_mse_std" if "val_mse_std" in rows[0] else None
    return (float(rows[0]["train_loss"]), float(rows[-1]["train_loss"]),
            float(rows[0][col]) if col else math.nan, float(rows[-1][col]) if col else math.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=pathlib.Path)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    c_ref = metrics.reference_constant(a.processed, "val")
    print(f"reference constant (best on val) = {c_ref:+.4f} deg")
    done = skipped = mismatched = 0
    labels = {}                       # split -> set of label/validity fingerprints
    groups = {}                       # (config key, split) -> {seed: record}
    predict0 = {}                     # split -> per-bin MAE of predicting 0 (labels only)
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
            labels.setdefault(split, set()).add(hash((y.tobytes(), v.tobytes())))
            predict0.setdefault(split, {b: r["mae_predict0"] for b, r in s["bins"].items()})
            key, seed = config_key(npz.parent)
            g = groups.setdefault((key, split), {})
            if seed not in g:                                   # first root listed wins
                ratio = float(np.std(p[v]) / np.std(y[v])) if np.std(y[v]) > 0 else float("nan")
                g[seed] = dict(macro=s["macro_mae"], ratio=ratio,
                               bins={b: r["mae_model"] for b, r in s["bins"].items()},
                               curve=first_last(npz.parent))
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
    print("distinct label/validity arrays per split:", {k: len(v) for k, v in sorted(labels.items())})
    for split, bins in sorted(predict0.items()):
        print(f"  predict-0 [{split}] bins " + " ".join(f"{b} {m:.3f}" for b, m in bins.items()))
    for (key, split), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rs = list(g.values())
        bins = {b: float(np.mean([r["bins"][b] for r in rs])) for b in rs[0]["bins"]}
        print(f"  {' '.join(key)} [{split}] n={len(rs)}  macro MAE {np.mean([r['macro'] for r in rs]):.3f}  "
              f"near-constant {sum(r['ratio'] < 0.01 for r in rs)}  bins " +
              " ".join(f"{b} {m:.3f}" for b, m in bins.items()))
        cur = np.array([r["curve"] for r in rs if r["curve"]])
        if split == "val" and len(cur):
            med = np.median(cur, axis=0)
            print(f"      median train loss {med[0]:.3f} -> {med[1]:.3f}, val MSE {med[2]:.3f} -> {med[3]:.3f}"
                  f" (first -> last epoch, {len(cur)} runs)")
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
