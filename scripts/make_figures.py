#!/usr/bin/env python3
"""Regenerate every README figure from runs/ with one command.

    python scripts/make_figures.py --runs runs --split test

The hard rule from the protocol: if this script does not exist, do not start launching
experiments. Assembling several hundred run directories by hand in the final week is how a
project with good runs ships a bad README.
"""

import argparse
import csv
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from metrics import BIN_NAMES

STYLE = {"cfc": ("C0", "-"), "lstm": ("C1", "--"), "lstm_dt": ("C1", "-."),
         "gru": ("C2", ":"), "cnn_mlp": ("0.45", "-"), "cnn_linear": ("0.65", "--"),
         "cnn_2frame": ("0.5", "-."), "transformer": ("C4", "-")}


def style(arm):
    return STYLE.get(arm, ("C3", "-"))


def training_curves(runs, out):
    """Validation binned MAE -- the SAME quantity early stopping selects on. Plotting the
    training loss instead would show a curve that connects to nothing in the results tables."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    any_ = False
    for d in sorted(pathlib.Path(runs).glob("*/metrics.csv")):
        cfg = json.loads((d.parent / "config.json").read_text())
        rows = list(csv.DictReader(d.open()))
        if not rows:
            continue
        any_ = True
        c, ls = style(cfg["arm"])
        ep = [int(r["epoch"]) for r in rows]
        a1.plot(ep, [float(r["train_loss"]) for r in rows], color=c, ls=ls, lw=1.2,
                label=f"{cfg['arm']} s{cfg['seed']}")
        a2.plot(ep, [float(r["val_curve_mae"]) for r in rows], color=c, ls=ls, lw=1.2)
    if not any_:
        plt.close(fig); return None
    a1.set_xlabel("epoch"); a1.set_ylabel("train loss (masked MSE, standardised)")
    a2.set_xlabel("epoch"); a2.set_ylabel("validation curve-bin MAE (deg)")
    a2.set_title("the quantity selection uses")
    a1.legend(fontsize=7, ncol=2)
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.suptitle("Training curves. With augmentation active, train loss sits ABOVE validation "
                 "for much of training;\nper-epoch validation is windowed and includes "
                 "cold-start frames, so it reads worse than the final rollout.", fontsize=8)
    fig.tight_layout()
    p = out / "training_curves.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def position_curve(runs, split, out):
    """Per-frame MAE against position within the evaluation window -- the headline figure.

    Non-recurrent arms must come out flat. That is the figure's own self-test: a sloped line
    for a single-frame CNN means the position accounting is broken, not that it grew memory.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    byarm, any_ = {}, False
    for f in sorted(pathlib.Path(runs).glob(f"*/final_metrics_{split}.json")):
        m = json.loads(f.read_text())
        c = m.get("mae_by_window_position")
        if c and c[0] is not None:
            byarm.setdefault(m["arm"], []).append(c)
            any_ = True
    if not any_:
        plt.close(fig); return None
    for arm, curves in sorted(byarm.items()):
        a = np.array(curves, dtype=float)
        c, ls = style(arm)
        t = np.arange(1, a.shape[1] + 1)
        ax.plot(t, np.median(a, 0), color=c, ls=ls, lw=1.8,
                label=f"{arm} (n={len(curves)})")
        if len(curves) > 1:
            ax.fill_between(t, a.min(0), a.max(0), color=c, alpha=0.15)
    ax.axvspan(0.5, 3.5, color="0.85", alpha=0.6, zorder=0)
    ax.text(2, ax.get_ylim()[1], "cold start", ha="center", va="top", fontsize=8, color="0.4")
    ax.set_xlabel("position within the evaluation window")
    ax.set_ylabel("per-frame MAE (deg)")
    ax.set_title("A flat line is correct for a non-recurrent arm", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "mae_by_window_position.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def bin_bars(runs, split, out):
    """Per-bin MAE against predict-0, which is the comparison the marking scheme asks for."""
    rows = [json.loads(f.read_text())
            for f in sorted(pathlib.Path(runs).glob(f"*/final_metrics_{split}.json"))]
    if not rows:
        return None
    arms, base = {}, rows[0]["rollout"]["bins"]
    for r in rows:
        arms.setdefault(r["arm"], []).append(r["rollout"]["bins"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(BIN_NAMES))
    w = 0.8 / (len(arms) + 1)
    ax.bar(x, [base[b]["mae_predict0"] for b in BIN_NAMES], w, label="predict-0",
           color="0.3")
    for i, (arm, bs) in enumerate(sorted(arms.items()), start=1):
        vals = [np.median([b[n]["mae_model"] for b in bs]) for n in BIN_NAMES]
        ax.bar(x + i * w, vals, w, label=arm, color=style(arm)[0])
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels([f"{n}\nn={base[n]['n_frames']}, {base[n]['n_events']} ev"
                        for n in BIN_NAMES], fontsize=8)
    ax.set_ylabel("MAE (deg)"); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    ax.set_title(f"Per-bin MAE, {split} split, against the constant-zero baseline", fontsize=9)
    fig.tight_layout()
    p = out / "bin_mae.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    made = [training_curves(args.runs, args.out),
            position_curve(args.runs, args.split, args.out),
            bin_bars(args.runs, args.split, args.out)]
    made = [m for m in made if m]
    if not made:
        print(f"nothing to plot: no runs under {args.runs}/", file=sys.stderr)
        raise SystemExit(1)
    for m in made:
        print(f"wrote {m}")


if __name__ == "__main__":
    main()
