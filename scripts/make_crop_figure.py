#!/usr/bin/env python3
"""Render the field-of-view comparison behind the crop experiment.

`data/preprocess.py` keeps the bottom SRC_CROP_ROWS rows of each 455x256 source frame and
resizes that band into the stored 66x240. The default, 150, comes from the reference
loader's `[-150:]`. This figure asks what that band actually contains.

The output PNG is deliberately NOT committed: .gitignore excludes *.png because the panels
are dataset frames, and the repo must not redistribute the data. The figure is reproducible
from this script, which is what the README cites.

    python scripts/make_crop_figure.py --data-root data/raw --processed data/processed \
        --out figures/crop_comparison.png
"""
import argparse, pathlib
import numpy as np
from PIL import Image, ImageDraw

ROWS = [150, 200, 256]
LINE_COLOURS = [(255, 0, 0), (0, 200, 255), (0, 255, 0)]


def find_images(root):
    imgs = sorted(root.glob("*.jpg"))
    if imgs:
        return root, imgs
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = sorted(d.glob("*.jpg"))
        if imgs:
            return d, imgs
    raise SystemExit(f"no .jpg under {root}")


def panel(im, label_deg):
    W, H = im.size
    full = im.copy()
    d = ImageDraw.Draw(full)
    for r, c in zip(ROWS, LINE_COLOURS):
        d.line([(0, H - r), (W, H - r)], fill=c, width=2)
    tiles = [full.resize((455, 256))]
    for r in ROWS:
        band = im.crop((0, max(0, H - r), W, H)).resize((240, 66), Image.BILINEAR)
        tiles.append(band.resize((455, 125), Image.NEAREST))
    strip = Image.new("RGB", (455, sum(t.height for t in tiles) + 10 * len(tiles)), (20, 20, 20))
    y = 0
    for t in tiles:
        strip.paste(t, (0, y)); y += t.height + 10
    return strip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=pathlib.Path, default=pathlib.Path("data/raw"))
    ap.add_argument("--processed", type=pathlib.Path, default=pathlib.Path("data/processed"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("figures/crop_comparison.png"))
    a = ap.parse_args()

    _, imgs = find_images(a.data_root)
    ang = np.load(a.processed / "angles_deg.npy")[:len(imgs)]
    picks = [int(np.argmax(np.abs(ang))), int(np.argmin(np.abs(ang)))]

    panels = [panel(Image.open(imgs[i]).convert("RGB"), ang[i]) for i in picks]
    out = Image.new("RGB", (sum(p.width for p in panels) + 20,
                            max(p.height for p in panels)), (20, 20, 20))
    x = 0
    for p in panels:
        out.paste(p, (x, 0)); x += p.width + 20
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(a.out)
    print(f"wrote {a.out}  frames #{picks[0]} ({ang[picks[0]]:+.0f} deg) "
          f"and #{picks[1]} ({ang[picks[1]]:+.0f} deg)")
    print("rows, top to bottom: source with crop lines, then the "
          + " / ".join(str(r) for r in ROWS) + " crops as the model receives them")


if __name__ == "__main__":
    main()
