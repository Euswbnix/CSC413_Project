#!/usr/bin/env python3
"""Evaluate a trained run and write its reported numbers.

    python evaluate.py runs/cfc_s0_T16_lr0.001_k0.1_aug-full --split test

Two evaluation paths, deliberately different, both reported:

**Stateful rollout** over contiguous segments, carrying `hx` between 256-frame chunks. This
is the FINAL-NUMBER path: it covers every frame in the split exactly once, with no double
counting and none discarded. Equivalent to one long forward pass -- asserted in
`tests/test_dataset_windows.py`.

**Windowed**, state reset at every window start, stride 1. Every frame is therefore scored at
all T positions, which makes the per-position comparison WITHIN-frame and gives the headline
figure. It reads slightly worse than the rollout because it includes cold-start frames; say so
under the figure or it looks like a bug.

The validity mask is applied in both. 973 frames carry a logging default of 0.0 in place of a
real angle; scoring them would flatter predict-0 in exactly the bin where it is strongest.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import metrics
from data.dataset import SteeringData, rollout_chunks, window_batches
from models.interface import build_arm


@torch.no_grad()
def rollout_predictions(model, data, split, chunk, fixed_dt):
    """Returns (pred_deg, true_deg, valid) covering the split exactly once, in order."""
    model.eval()
    P, Y, V, hx, seg = [], [], [], None, None
    for (seg_i, s, e), b in rollout_chunks(data, split, chunk=chunk):
        if seg_i != seg:           # a new segment: the state must not cross the gap
            hx, seg = None, seg_i
        dt = torch.ones_like(b.dt) if fixed_dt else b.dt
        pred, hx = model(b.frames, dt=dt, hx=hx)
        hx = hx.detach() if torch.is_tensor(hx) else hx
        P.append(pred.squeeze(-1).squeeze(0).float().cpu())
        Y.append(b.y.squeeze(0).float().cpu())
        V.append(b.valid.squeeze(0).cpu())
    p = data.to_degrees(torch.cat(P)).numpy()
    y = data.to_degrees(torch.cat(Y)).numpy()
    return p, y, torch.cat(V).numpy()


@torch.no_grad()
def windowed_predictions(model, data, split, T, batch_size):
    """(n_windows, T) arrays with the state reset at every window start."""
    model.eval()
    P, Y, V = [], [], []
    for b in window_batches(data, split, T, batch_size):
        pred, _ = model(b.frames, dt=b.dt)
        P.append(pred.squeeze(-1).float().cpu())
        Y.append(b.y.float().cpu())
        V.append(b.valid.cpu())
    p = data.to_degrees(torch.cat(P)).numpy()
    y = data.to_degrees(torch.cat(Y)).numpy()
    return p, y, torch.cat(V).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=pathlib.Path)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-predictions", action="store_true")
    args = ap.parse_args()

    cfg = json.loads((args.run / "config.json").read_text())
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data = SteeringData(args.processed, device=dev, pin=False)
    model = build_arm(cfg["arm"]).to(dev)
    ck = torch.load(args.run / "checkpoints" / args.checkpoint, map_location=dev)
    model.load_state_dict(ck["model"])

    p, y, v = rollout_predictions(model, data, args.split, args.chunk, cfg.get("fixed_dt", False))
    s = metrics.summary(p, y, v)

    T = cfg["T"]
    wp, wy, wv = windowed_predictions(model, data, args.split, T, 64)
    curve = metrics.mae_by_window_position(wp, wy, wv, T)

    # The same weights, evaluated with the state severed before every step. No retraining:
    # the gap is the contribution of carrying state at all, isolated from everything else.
    out = {
        "run": args.run.name, "arm": cfg["arm"], "seed": cfg["seed"], "split": args.split,
        "checkpoint": args.checkpoint, "best_epoch": ck.get("epoch"),
        "n_frames": int(len(y)), "rollout": s,
        "mae_by_window_position": [None if np.isnan(x) else float(x) for x in curve],
        "position_delta_deg": float(curve[0] - curve[-1]),
        "params": cfg["arm_params"],
    }
    (args.run / f"final_metrics_{args.split}.json").write_text(json.dumps(out, indent=2) + "\n")
    if args.save_predictions:
        np.savez_compressed(args.run / f"predictions_{args.split}.npz",
                            pred=p, true=y, valid=v)

    b = s["bins"]
    print(f"\n{args.run.name}  [{args.split}]  n={len(y)}")
    print(f"  macro skill {s['macro_skill']:+.4f}   macro MAE {s['macro_mae']:.2f} deg"
          f"  (predict-0 {s['macro_mae_predict0']:.2f})")
    print(f"  global MAE  {s['global_mae']:.2f}   Pearson r {s['pearson_r']:+.3f}"
          f"   false alarm {s['false_alarm_rate']:.3f}")
    print(f"  {'bin':18} {'n':>7} {'ev':>4} {'model':>8} {'pred-0':>8} {'persist':>8} {'ratio':>7}")
    for name, r in b.items():
        ratio = r["mae_model"] / r["mae_predict0"] if r["mae_predict0"] else float("nan")
        print(f"  {name:18} {r['n_frames']:>7} {r['n_events']:>4} {r['mae_model']:>8.2f}"
              f" {r['mae_predict0']:>8.2f} {r['mae_persistence']:>8.2f} {ratio:>7.3f}")
    print(f"  per-position MAE: t=1 {curve[0]:.2f} -> t={T} {curve[-1]:.2f}"
          f"  (delta {curve[0]-curve[-1]:+.2f} deg)")


if __name__ == "__main__":
    main()
