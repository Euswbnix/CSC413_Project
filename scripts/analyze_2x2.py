#!/usr/bin/env python3
"""Analyse the {feature LayerNorm} x {encoder lr} mechanism screen (scripts/run_2x2.sh).

DECISION RULES -- written before any result of this experiment was seen, validation only:

1. Mechanism. Relative to the SAME SEED in the untreated cell, does a treatment lower the
   first-step sigmoid saturation and the feature RMS at the selected (best) epoch? Counted as an
   effect only if all three seeds move the same way.
2. Performance. Does the cell's mean validation macro MAE beat the best constant on validation,
   with no seed near-constant (output std / label std >= 0.01)?
3. n = 3 per cell is a screen. Per-seed paired differences are reported; no p-values.

Test is not read. Paths are relative to the project root; --root points at the 2x2 runs.
"""
import argparse, csv, json, math, pathlib
import numpy as np

CELLS = {"base": "", "fn": "_fn", "elr": "_elr0.0001", "fn_elr": "_fn_elr0.0001"}
NEAR_CONSTANT = 0.01


def read_csv(p):
    with open(p) as fh:
        return list(csv.DictReader(fh))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def run(root, cell, seed):
    d = pathlib.Path(root) / f"cfc_s{seed}_T16_lr0.001_k0.1_aug-full{CELLS[cell]}"
    fm = json.loads((d / "final_metrics.json").read_text())
    rv = json.loads((d / "final_metrics_val.json").read_text())["rollout"]
    z = np.load(d / "predictions_val.npz")
    p, y, v = z["pred"].astype(float), z["true"].astype(float), z["valid"].astype(bool)
    h = {int(float(r["epoch"])): r for r in read_csv(d / "health.csv")}
    be = fm["best_epoch"]
    last = max(h)
    return dict(
        cell=cell, seed=seed, best_epoch=be, epochs_run=last + 1,
        val_macro_mae=rv["macro_mae"], val_const_mae=rv.get("macro_mae_ref_constant"),
        val_ccc=rv.get("ccc"), val_r=rv.get("pearson_r"),
        std_ratio=float(p[v].std() / y[v].std()),
        bins={k: b["mae_model"] for k, b in rv["bins"].items() if not k.startswith("(")},
        rms_init=f(h[-1]["feature_rms"]), rms_best=f(h[be]["feature_rms"]),
        rms_last=f(h[last]["feature_rms"]),
        sat_best=f(h[be]["sigmoid_saturated_fraction"]),
        sat_last=f(h[last]["sigmoid_saturated_fraction"]),
        clip_mean=float(np.nanmean([f(h[e]["clip_fraction"]) for e in h if e >= 0])),
        tr_clean=(f(h[-1]["train_clean_mse_std"]), f(h[be]["train_clean_mse_std"]),
                  f(h[last]["train_clean_mse_std"])),
        val_mse=(f(h[-1]["val_mse_std"]), f(h[be]["val_mse_std"]), f(h[last]["val_mse_std"])),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_2x2")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    R = {(c, s): run(a.root, c, s) for c in CELLS for s in seeds}
    const = np.nanmean([r["val_const_mae"] for r in R.values()])

    print(f"validation only. best constant (fit on val) macro MAE = {const:.3f}; "
          f"predict-0 = 16.44\n")
    print(f"{'cell':<8}{'seed':>5}{'best_ep':>8}{'ran':>5}{'val MAE':>9}{'CCC':>8}{'r':>8}"
          f"{'std比':>8}{'RMS init→best→last':>24}{'饱和 best/last':>16}{'clip':>7}")
    for c in CELLS:
        for s in seeds:
            r = R[(c, s)]
            print(f"{c:<8}{s:>5}{r['best_epoch']:>8}{r['epochs_run']:>5}{r['val_macro_mae']:>9.3f}"
                  f"{r['val_ccc']:>+8.3f}{r['val_r']:>+8.3f}{r['std_ratio']:>8.3f}"
                  f"{r['rms_init']:>8.2f}→{r['rms_best']:>6.2f}→{r['rms_last']:>6.2f}"
                  f"{r['sat_best']:>8.2f}/{r['sat_last']:.2f}{r['clip_mean']:>7.2f}")
        print()

    print("cell means (n=3):")
    print(f"{'cell':<8}{'val MAE':>9}{'vs const':>10}{'CCC':>8}{'near-const':>12}"
          f"{'straight':>10}{'gentle':>8}{'curve':>8}")
    for c in CELLS:
        rs = [R[(c, s)] for s in seeds]
        m = np.mean([r["val_macro_mae"] for r in rs])
        b = {k: np.mean([r["bins"][k] for r in rs]) for k in rs[0]["bins"]}
        print(f"{c:<8}{m:>9.3f}{m - const:>+10.3f}{np.mean([r['val_ccc'] for r in rs]):>+8.3f}"
              f"{sum(r['std_ratio'] < NEAR_CONSTANT for r in rs):>9}/{len(rs)}"
              + "".join(f"{v:>9.2f}" for v in b.values()))

    print("\nrule 1 -- paired per seed vs base (negative = lower than base):")
    for c in ("fn", "elr", "fn_elr"):
        dsat = [R[(c, s)]["sat_best"] - R[("base", s)]["sat_best"] for s in seeds]
        drms = [R[(c, s)]["rms_best"] - R[("base", s)]["rms_best"] for s in seeds]
        dmae = [R[(c, s)]["val_macro_mae"] - R[("base", s)]["val_macro_mae"] for s in seeds]
        same = lambda xs: all(x < 0 for x in xs) or all(x > 0 for x in xs)
        print(f"  {c:<7} Δ饱和 {['%+.2f' % x for x in dsat]} {'一致' if same(dsat) else '不一致'}   "
              f"ΔRMS {['%+.2f' % x for x in drms]} {'一致' if same(drms) else '不一致'}   "
              f"Δval MAE {['%+.3f' % x for x in dmae]} {'一致' if same(dmae) else '不一致'}")

    print("\nrule 2 -- beats the val constant with no near-constant seed:")
    for c in CELLS:
        rs = [R[(c, s)] for s in seeds]
        m = np.mean([r["val_macro_mae"] for r in rs])
        ok = m < const and not any(r["std_ratio"] < NEAR_CONSTANT for r in rs)
        print(f"  {c:<7} {'PASS' if ok else 'fail'}  (mean {m:.3f} vs {const:.3f})")

    print("\nclean-train vs val MSE (standardised) at init → best → last, per cell mean:")
    for c in CELLS:
        rs = [R[(c, s)] for s in seeds]
        tr = np.mean([r["tr_clean"] for r in rs], axis=0)
        va = np.mean([r["val_mse"] for r in rs], axis=0)
        print(f"  {c:<7} train-clean {tr[0]:.3f}→{tr[1]:.3f}→{tr[2]:.3f}    "
              f"val {va[0]:.3f}→{va[1]:.3f}→{va[2]:.3f}")
    out = pathlib.Path("results"); out.mkdir(exist_ok=True)
    (out / "analysis_2x2.json").write_text(json.dumps(
        {f"{c}_s{s}": r for (c, s), r in R.items()}, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
