#!/usr/bin/env python3
"""Do the models over-steer, and is that inherited from the training distribution?

The chronological split leaves train 2x curvier than val/test (27.9% curve frames against
12.0% and 14.4%; |y| mean 15.82 against 8.40 and 8.22). A model fit to minimise squared
error on train therefore has a prior over output magnitude that the evaluation splits do
not share. If that is what is happening, predictions should be systematically WIDER than
the labels on val and test -- and the excess should land in the straight bin, which is 55-61%
of evaluation frames and the one bin where predict-0 is close to unbeatable.

Reads the prediction arrays `evaluate.py --save-predictions` writes, so it measures the
reported predictions rather than re-deriving them.
"""
import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "lab")
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "val"
ARMS = ("cnn_linear", "cnn_2frame", "cnn_mlp", "cnn_avg", "lstm_dt", "lstm", "cfc", "gru")


def arm_of(n):
    return next((a for a in ARMS if n.startswith(a)), n.split("_")[0])


rows = {}
for f in ROOT.rglob(f"predictions_{SPLIT}.npz"):
    z = np.load(f)
    p, y, v = z["pred"], z["true"], z["valid"].astype(bool)
    p, y = p[v], y[v]
    st = np.abs(y) < 5
    rows.setdefault(arm_of(f.parent.name), []).append(dict(
        pred_std=p.std(), true_std=y.std(),
        pred_absmean=np.abs(p).mean(), true_absmean=np.abs(y).mean(),
        straight_pred_absmean=np.abs(p[st]).mean(), straight_true_absmean=np.abs(y[st]).mean(),
        straight_bias=np.abs(p[st]).mean() - np.abs(y[st]).mean(),
        frac_pred_over15=(np.abs(p) >= 15).mean(), frac_true_over15=(np.abs(y) >= 15).mean(),
    ))

if not rows:
    print(f"no predictions_{SPLIT}.npz under {ROOT}"); sys.exit()

hdr = (f"{'arm':<11}{'n':>4}{'pred std':>10}{'true std':>10}{'比值':>7}"
       f"{'直行帧 |pred|':>14}{'直行帧 |true|':>14}{'过量':>8}{'预测>15%':>10}{'真实>15%':>10}")
print(f"[{SPLIT}]  " + hdr); print("-" * (len(hdr) + 7))
for a in sorted(rows):
    rs = rows[a]
    m = lambda k: float(np.median([r[k] for r in rs]))
    print(f"{'':<7}{a:<11}{len(rs):>4}{m('pred_std'):>10.2f}{m('true_std'):>10.2f}"
          f"{m('pred_std')/m('true_std'):>7.2f}{m('straight_pred_absmean'):>14.2f}"
          f"{m('straight_true_absmean'):>14.2f}{m('straight_bias'):>+8.2f}"
          f"{m('frac_pred_over15')*100:>10.1f}{m('frac_true_over15')*100:>10.1f}")
print("\n直行帧 = |真实角| < 5 度的帧。「过量」= 模型在这些帧上输出的平均幅度 − 真实平均幅度。")
print("过量为正 = 在本该几乎不打方向的地方打了方向，这正是 predict-0 赢的地方。")
