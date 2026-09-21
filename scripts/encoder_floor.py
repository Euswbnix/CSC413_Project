#!/usr/bin/env python3
"""The noise floor AS THE MODEL SEES IT.

scripts/noise_floor.py compares frames with a coarse grey descriptor, which answers a
different question: two frames a 17x60 thumbnail cannot separate may be perfectly separable
at 66x240x3. That measurement bounds the floor from above and cannot settle it.

This one uses a TRAINED ENCODER's own 32-d features as the descriptor. If two frames the
encoder maps to nearly the same point carry very different steering angles, then that
disagreement is irreducible FOR THIS MODEL -- the representation has thrown the distinction
away, and no amount of recurrent machinery downstream can recover it.

Read the two together. If the floor stays high under the encoder's own features, the target
is genuinely ambiguous given the input. If it drops sharply, the coarse measurement was an
artefact of resolution and the headroom is real.
"""

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from models.interface import build_arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=pathlib.Path, help="a trained run directory")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--min-sep", type=int, default=2000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import json
    cfg = json.loads((args.run / "config.json").read_text())
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    m = build_arm(cfg["arm"]).to(dev).eval()
    m.load_state_dict(torch.load(args.run / "checkpoints" / "best.pt",
                                 map_location=dev)["model"])

    d = pathlib.Path(args.processed)
    frames = np.load(d / "frames.npy", mmap_mode="r")
    ang = np.load(d / "angles_deg.npy").astype(np.float64)
    valid = ~np.load(d / "label_dropout.npy")
    idx = np.arange(0, len(ang), args.stride)
    idx = idx[valid[idx]]

    off = (frames.shape[2] - 200) // 2
    feats = []
    with torch.no_grad():
        for a in range(0, len(idx), 256):
            b = np.asarray(frames[idx[a:a + 256]][:, :, off:off + 200], dtype=np.float32) / 255.0
            t = torch.from_numpy(b).permute(0, 3, 1, 2).unsqueeze(1).to(dev)
            feats.append(m.encoder(t).squeeze(1).cpu())
    F = torch.cat(feats).numpy().astype(np.float32)
    F -= F.mean(0, keepdims=True)
    F /= (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    print(f"encoder features for {len(idx)} frames, from {args.run.name}")

    sep = args.min_sep // args.stride
    print(f"\n  {'cos >':>7} {'n pairs':>9} {'median':>8} {'mean':>8}  "
          f"{'on curves: n':>13} {'mean':>7} {'=> MAE bound':>13}")
    for thr in (0.90, 0.95, 0.98, 0.99, 0.995):
        pi, pj = [], []
        for a in range(0, len(idx), 512):
            blk = F[a:a + 512] @ F.T
            for r in range(blk.shape[0]):
                i = a + r
                j = np.flatnonzero((blk[r] > thr) & (np.arange(len(idx)) > i + sep))
                pi.extend([i] * len(j)); pj.extend(j.tolist())
        if len(pi) < 5:
            print(f"  {thr:>7.3f} {len(pi):>9}   (too few)"); continue
        pi = np.array(pi); pj = np.array(pj)
        dis = np.abs(ang[idx[pi]] - ang[idx[pj]])
        big = np.abs(ang[idx[pi]]) >= 15
        cm = dis[big].mean() if big.any() else float("nan")
        print(f"  {thr:>7.3f} {len(pi):>9} {np.median(dis):>8.2f} {dis.mean():>8.2f}"
              f"  {int(big.sum()):>13} {cm:>7.2f} {cm/2:>13.2f}")
    print("\n  The MAE bound is the curve-pair disagreement halved: a model predicting the")
    print("  midpoint of two frames it cannot distinguish errs by half their gap on each.")
    print("  Our best curve-bin MAE is 25.1; predict-0 is 28.86.")


if __name__ == "__main__":
    main()
