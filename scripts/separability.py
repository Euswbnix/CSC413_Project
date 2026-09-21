#!/usr/bin/env python3
"""Can the model tell a straight frame from a curved one at all?

Re-weighting the bins so the straight bin carried a third of the gradient did not shrink
what the model emits on straight frames -- it grew, 6.89 -> 7.41 deg. So the straight-bin
error is not a gradient-budget problem. The remaining explanation is discriminability: if
the encoder cannot separate the two classes of frame, no objective over those frames can
fix it, because the model has nothing to condition on.

Measured as AUC of |pred| ranking curve frames (|y|>=15) above straight frames (|y|<5).
AUC 0.5 = the prediction magnitude carries no information about which kind of frame it is;
1.0 = perfectly separated. Reported beside the same AUC for |y| itself, which is 1.0 by
construction, and for a few named baselines.
"""
import pathlib, sys
import numpy as np

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "lab")
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "val"
ARMS = ("cnn_linear", "cnn_2frame", "cnn_mlp", "cnn_avg", "lstm_dt", "lstm", "cfc", "gru")


def auc(score, pos):
    """P(score[pos] > score[neg]), ties counted as half. Rank form, no sklearn."""
    s = np.asarray(score, dtype=np.float64)
    r = np.empty(len(s))
    order = np.argsort(s, kind="mergesort")
    sr = s[order]
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    np_, nn = pos.sum(), (~pos).sum()
    if np_ == 0 or nn == 0:
        return float("nan")
    return float((r[pos].sum() - np_ * (np_ + 1) / 2.0) / (np_ * nn))


rows = {}
for f in ROOT.rglob(f"predictions_{SPLIT}.npz"):
    z = np.load(f)
    p, y, v = z["pred"], z["true"], z["valid"].astype(bool)
    p, y = p[v], y[v]
    keep = (np.abs(y) < 5) | (np.abs(y) >= 15)
    p, y = p[keep], y[keep]
    pos = np.abs(y) >= 15
    arm = next((a for a in ARMS if f.parent.name.startswith(a)), f.parent.name.split("_")[0])
    rows.setdefault(arm, []).append((auc(np.abs(p), pos), auc(p * np.sign(y), pos)))

if not rows:
    print(f"no predictions_{SPLIT}.npz under {ROOT}"); sys.exit()
print(f"[{SPLIT}]  直行(|y|<5) vs 急弯(|y|>=15) 的可分性\n")
print(f"{'arm':<12}{'n':>4}{'AUC(|pred|)':>13}{'AUC(有向)':>12}")
print("-" * 41)
for a in sorted(rows):
    v = np.array(rows[a], dtype=float)
    print(f"{a:<12}{len(v):>4}{np.median(v[:, 0]):>13.3f}{np.median(v[:, 1]):>12.3f}")
print("\nAUC 0.5 = 预测幅度完全无法区分直路与弯路; 1.0 = 完全分开。")
print("AUC(有向) 用 pred*sign(真实) 打分, 把方向也算进去。")
