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

    pairs, sep_min = [], args.min_sep // args.stride
    for a in range(0, len(idx), 256):
        blk = D[a:a + 256] @ D.T
        for r in range(blk.shape[0]):
            i = a + r
            j = np.flatnonzero((blk[r] > args.sim) & (np.abs(np.arange(len(idx)) - i) > sep_min))
            for k in j[j > i]:
                pairs.append((i, k))
    if not pairs:
        print(f"no pairs above similarity {args.sim} separated by {args.min_sep} frames")
        return
    pi = np.array([p[0] for p in pairs]); pj = np.array([p[1] for p in pairs])
    dis = np.abs(ang[idx[pi]] - ang[idx[pj]])
    moving = ~(still[pi] | still[pj])

    print(f"\n{len(pairs)} near-duplicate pairs (similarity > {args.sim}, "
          f">= {args.min_sep} frames apart)")
    for name, m in (("all pairs", np.ones(len(dis), bool)), ("both frames moving", moving)):
        if not m.any():
            continue
        v = dis[m]
        print(f"  {name:20} n={m.sum():>6}  |dangle| median {np.median(v):6.2f}  "
              f"mean {v.mean():6.2f}  p90 {np.percentile(v,90):6.2f}  max {v.max():6.2f}")

    big = np.abs(ang[idx[pi]]) >= 15
    if (big & moving).any():
        v = dis[big & moving]
        print(f"  {'pairs on a curve':20} n={(big&moving).sum():>6}  "
              f"|dangle| median {np.median(v):6.2f}  mean {v.mean():6.2f}")
        print(f"\n  => an image-only model cannot do better than about {v.mean()/2:.1f} deg MAE")
        print(f"     on curve frames, because frames it cannot tell apart disagree by that much.")
        print(f"     Our best curve-bin MAE so far: 25.1 deg (predict-0: 28.86).")


if __name__ == "__main__":
    main()
