#!/usr/bin/env python3
"""How much of the target is predictable from the image at all?

The question every other decision waits on. If two frames the camera cannot tell apart
carry very different steering angles, then no architecture recovers the difference, and a
curve-bin MAE near that disagreement is not a weak model -- it is the task's ceiling.

Method: find pairs of frames that are near-identical in appearance but far apart in the
recording (so they are not the same corner one frame later), and measure how much their
LABELS disagree. That disagreement is a lower bound on achievable error for any model that
sees only the image.

Two confounds are handled explicitly. Pairs close in time are excluded, or the measurement
just rediscovers that steering is smooth. And likely-stationary frames are reported
separately: a stopped car produces identical frames whose labels agree trivially, which
would drag the floor down for the wrong reason.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--min-sep", type=int, default=2000, help="raw frames apart, minimum")
    ap.add_argument("--sim", type=float, default=0.95)
    args = ap.parse_args()

    d = pathlib.Path(args.processed)
    frames = np.load(d / "frames.npy", mmap_mode="r")
    ang = np.load(d / "angles_deg.npy").astype(np.float64)
    valid = ~np.load(d / "label_dropout.npy") if (d / "label_dropout.npy").exists() \
        else np.ones(len(ang), bool)

    idx = np.arange(0, len(ang), args.stride)
    idx = idx[valid[idx]]
    print(f"descriptors for {len(idx)} frames (every {args.stride})")
    D = np.asarray(frames[idx][:, ::4, ::4].mean(axis=3), dtype=np.float32).reshape(len(idx), -1)
    D -= D.mean(1, keepdims=True)
    D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)

    # interframe motion, as a stationary proxy
    mv = np.concatenate([[np.inf], np.abs(np.diff(D, axis=0)).mean(1)])
    still = mv < np.percentile(mv[np.isfinite(mv)], 10)

    sep_min = args.min_sep // args.stride
    print(f"\npair label disagreement as a function of appearance similarity")
    print(f"(pairs at least {args.min_sep} raw frames apart; a single threshold gives too few")
    print(f" pairs to be a measurement, so the TREND toward high similarity is the estimate)\n")
    print(f"  {'sim >':>7} {'n pairs':>9} {'median':>8} {'mean':>8} {'p90':>8}  {'on curves: n':>13} {'mean':>7}")
    for thr in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        pi, pj = [], []
        for a in range(0, len(idx), 256):
            blk = D[a:a + 256] @ D.T
            for r in range(blk.shape[0]):
                i = a + r
                j = np.flatnonzero((blk[r] > thr) & (np.arange(len(idx)) > i + sep_min))
                pi.extend([i] * len(j)); pj.extend(j.tolist())
        if len(pi) < 3:
            print(f"  {thr:>7.2f} {len(pi):>9}   (too few)"); continue
        pi = np.array(pi); pj = np.array(pj)
        moving = ~(still[pi] | still[pj])
        dis = np.abs(ang[idx[pi]] - ang[idx[pj]])[moving]
        big = (np.abs(ang[idx[pi]]) >= 15)[moving]
        cm = dis[big].mean() if big.any() else float("nan")
        print(f"  {thr:>7.2f} {int(moving.sum()):>9} {np.median(dis):>8.2f} {dis.mean():>8.2f}"
              f" {np.percentile(dis,90):>8.2f}  {int(big.sum()):>13} {cm:>7.2f}")
    print("\n  Read the last column down the table. If it keeps falling as similarity rises,")
    print("  the floor is below the last value; if it flattens, that plateau IS the floor.")
    print("  Divide by two for a per-frame MAE bound -- the disagreement is between two frames,")
    print("  and a model predicting their midpoint errs by half of it on each.")
    print("\n  For reference: our best curve-bin MAE is 25.1 deg, predict-0 is 28.86.")


if __name__ == "__main__":
    main()
