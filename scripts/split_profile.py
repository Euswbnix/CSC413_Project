#!/usr/bin/env python3
"""Label distribution per split, and what predict-0 scores on each.

The split effect (rollout-test minus rollout-val) is the largest single term for the CfC.
Before reading that as a generalisation failure, check the cheaper explanation: the splits
are chronological, so they are different stretches of road and need not carry the same
angle distribution at all. If predict-0's own macro MAE moves between splits, then part of
the "drop" is the yardstick moving, not the model.
"""
import json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import metrics

P = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed")
ang = np.load(P / "angles_deg.npy")
man = json.loads((P / "manifest.json").read_text())
drop = P / "label_dropout.npy"
bad = np.load(drop) if drop.exists() else np.zeros(len(ang), dtype=bool)
bad = bad.astype(bool)

splits = man.get("splits", man)
hdr = f"{'split':<7}{'n':>7}{'valid':>7}{'|y|mean':>9}{'|y|med':>8}{'std':>8}{'straight%':>10}{'gentle%':>9}{'curve%':>8}{'p0 macroMAE':>13}"
print(hdr); print("-" * len(hdr))
for s in ("train", "val", "test"):
    sp = splits[s]
    lo, hi = (sp["start"], sp["end"]) if isinstance(sp, dict) else (sp[0], sp[1])
    y = ang[lo:hi]
    v = ~bad[lo:hi]
    yv = y[v]
    ab = np.abs(yv)
    print(f"{s:<7}{len(y):>7}{v.sum():>7}{ab.mean():>9.2f}{np.median(ab):>8.2f}{yv.std():>8.2f}"
          f"{(ab < 5).mean()*100:>10.1f}{((ab >= 5) & (ab < 15)).mean()*100:>9.1f}"
          f"{(ab >= 15).mean()*100:>8.1f}{metrics.macro_mae(np.zeros_like(yv), yv):>13.2f}")
print("\np0 macroMAE = predict-0 的 macro MAE。它在各 split 间的差异就是「尺子本身」的移动。")
