#!/usr/bin/env python3
"""One decode pass: JPEG -> uint8 memmap, plus the manifest everything downstream reads.

Run AFTER data/eda_sullychen.py, because the split buffer is derived from the measured
decorrelation lag and the bin decision depends on gate G1.

Why it is shaped this way
-------------------------
* **`data.txt` line i IS memmap row i.** We parse `data.txt` in order and never glob the
  image directory. ~161 images in the archive have no label line (repo issue #2), so any
  positional join over the directory is silently misaligned -- and a uniform frame/label
  off-by-one passes every summary statistic in the EDA. `--verify-gif` renders a strip with
  the true angle overlaid, which is the only check that catches it.
* **Stored at 66x240, model input is 66x200.** The extra 40 px of width exists so the
  horizontal-shift augmentation has real pixels to shift into. Store at the model width and
  every shift must edge-pad, which makes the padding band's width a perfect linear function
  of the label correction -- the model can then regress the label off a border artifact and
  the augmentation ablation becomes meaningless. Costs 2.99 GB instead of 2.49 GB.
  DO NOT "optimise away" the 0.5 GB.
* **Crop the bottom 150 rows before resizing**, matching the reference loader's `[-150:]`.
  Cite that provenance in Data Transformation.
* **uint8, not float32.** The win is eliminating per-epoch JPEG decode, which is a CPU cost.
  It is not a VRAM decision -- float32 would also fit on a 32 GB card.
* **Segments, not just splits.** A 16-frame window that straddles a multi-second recording
  gap is a legal sample with a discontinuous label, so the index is cut at timestamp gaps and
  the sampler refuses to cross a SEGMENT boundary as well as a split boundary.
* **Normalisation statistics come from the TRAIN split only**, are written here once, and are
  read identically by train/evaluate/ood. Computing them per-run, or over all frames, leaks.

Usage:
    python data/preprocess.py --data-root data/raw --out data/processed
    python data/preprocess.py --data-root data/raw --out data/processed --verify-gif 1200
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from data.eda_sullychen import (parse_data_txt, chronological_splits, ACF_LAGS,  # noqa: E402
                                MIN_BUFFER, SPLIT_FRACTIONS)

SRC_CROP_ROWS = 150          # bottom rows of the 455x256 source, per the reference loader
STORE_H, STORE_W = 66, 240
MODEL_W = 200
GAP_SECONDS = 0.5


def decorrelation_buffer(angles):
    x = angles - angles.mean()
    denom = float((x * x).sum())
    for lag in ACF_LAGS:
        if lag < len(x) and abs(float((x[:-lag] * x[lag:]).sum() / denom)) < 0.1:
            return max(lag, MIN_BUFFER)
    return MIN_BUFFER


def segments_from_timestamps(stamps, n):
    """Contiguous runs with no gap larger than GAP_SECONDS. Without timestamps there is one
    segment and we say so, rather than assuming a frame rate."""
    if stamps is None:
        return [(0, n)], None
    t = np.array([s.timestamp() for s in stamps])
    dt = np.diff(t)
    bounds = [0, *(np.flatnonzero(dt > GAP_SECONDS) + 1).tolist(), n]
    segs = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    return segs, float(np.median(dt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/processed"))
    ap.add_argument("--verify-gif", type=int, default=None, metavar="START_FRAME",
                    help="render 10 s from this frame with the true angle overlaid. The ONLY "
                         "check that catches a uniform frame/label off-by-one. PICK A CURVY "
                         "STRETCH: on straight road the needle sits at zero and the check "
                         "proves nothing. Use the EDA's per-slice table to find one.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from PIL import Image, ImageDraw

    names, angles, stamps = parse_data_txt(args.data_root / "data.txt")
    n = len(names)
    print(f"{n} label lines -> memmap rows 0..{n-1} (line i IS row i)")

    frames = np.lib.format.open_memmap(args.out / "frames.npy", mode="w+", dtype=np.uint8,
                                       shape=(n, STORE_H, STORE_W, 3))
    missing = []
    for i, name in enumerate(names):
        p = args.data_root / name
        if not p.exists():
            missing.append(i)
            continue
        img = Image.open(p).convert("RGB")
        img = img.crop((0, img.height - SRC_CROP_ROWS, img.width, img.height))
        frames[i] = np.asarray(img.resize((STORE_W, STORE_H), Image.BILINEAR), dtype=np.uint8)
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{n}")
    frames.flush()
    if missing:
        # Rows whose image is absent are left zeroed AND recorded, so the sampler can skip
        # windows containing them. Silently training on black frames with a real label is
        # exactly the kind of thing that shows up as an unexplained architecture result.
        print(f"WARNING: {len(missing)} label lines had no image file; rows left zeroed "
              f"and recorded in the manifest as `missing_rows`")

    segs, median_dt = segments_from_timestamps(stamps, n)
    buffer = decorrelation_buffer(angles)
    splits = chronological_splits(n, buffer, SPLIT_FRACTIONS)
    tr_lo, tr_hi = splits["train"]

    # TRAIN-SPLIT ONLY. Computing these over all frames leaks test statistics into training.
    train_angles = angles[tr_lo:tr_hi]
    target_mean, target_std = float(train_angles.mean()), float(train_angles.std())
    sample = frames[tr_lo:tr_hi:max(1, (tr_hi - tr_lo) // 2000)].astype(np.float32) / 255.0
    pixel_mean = sample.mean(axis=(0, 1, 2)).tolist()
    pixel_std = sample.std(axis=(0, 1, 2)).tolist()

    manifest = {
        "n_frames": n,
        "stored_hw": [STORE_H, STORE_W],
        "model_w": MODEL_W,
        "src_crop_rows": SRC_CROP_ROWS,
        "median_dt_s": median_dt,
        "fps_measured": (1.0 / median_dt) if median_dt else None,
        "has_timestamps": stamps is not None,
        "gap_seconds": GAP_SECONDS,
        "segments": [[int(a), int(b)] for a, b in segs],
        "split_buffer": int(buffer),
        "split_fractions": list(SPLIT_FRACTIONS),
        "splits": {k: [int(a), int(b)] for k, (a, b) in splits.items()},
        "missing_rows": [int(i) for i in missing],
        "target_mean_deg": target_mean,
        "target_std_deg": target_std,
        "pixel_mean": pixel_mean,
        "pixel_std": pixel_std,
        "normalisation_scope": "train split only",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    np.save(args.out / "angles_deg.npy", angles.astype(np.float32))

    print(f"segments: {len(segs)}  buffer: {buffer}  splits: "
          + " ".join(f"{k}[{a},{b})" for k, (a, b) in splits.items()))
    print(f"target (train only): mean {target_mean:.3f} std {target_std:.3f} deg")
    print(f"wrote {args.out}/frames.npy "
          f"({n * STORE_H * STORE_W * 3 / 1e9:.2f} GB), angles_deg.npy, manifest.json")

    if args.verify_gif is not None and median_dt:
        k = int(round(10.0 / median_dt))
        lo = args.verify_gif
        hi = min(n, lo + k)
        out = []
        for i in range(lo, hi, 2):
            im = Image.fromarray(frames[i]).resize((STORE_W * 3, STORE_H * 3), Image.NEAREST)
            d = ImageDraw.Draw(im)
            d.text((6, 6), f"row {i}  {angles[i]:+.1f} deg", fill=(255, 255, 0))
            cx = STORE_W * 3 // 2
            d.line([(cx, STORE_H * 3 - 4),
                    (cx + int(angles[i] * 2.0), STORE_H * 3 - 34)], fill=(255, 0, 0), width=3)
            out.append(im)
        gif = args.out / f"verify_{lo}.gif"
        out[0].save(gif, save_all=True, append_images=out[1:], duration=66, loop=0)
        print(f"\nwrote {gif}. WATCH IT. The red needle must lean the same way the road "
              f"curves.\nThis is the only check in the project that catches a uniform "
              f"frame/label off-by-one; every EDA statistic passes such a misalignment.")


if __name__ == "__main__":
    main()
