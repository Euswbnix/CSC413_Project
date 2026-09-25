"""What would excluding the ROS / camera-clock disagreement cost? Prints aggregates only.

For every session, frames where ROS time strays from the camera clock by more than a threshold
(after a constant-rate fit, as in timestamp_gaps.py) are flagged. A target frame is lost if it is
flagged, or if any frame in its history window is flagged. Losses are counted over prediction
targets (moving > 3 m/s, valid steering label), overall, by |steer| bin, and by split of the
whole-session design A found by geo_split_feasibility.py (sessions without usable GPS: "none").
Usage (server): python hdd/drift_exclusion.py --raw ~/workspace/hdd/raw --checks ~/workspace/hdd/checks \
    --exclude 201706081335 201706081445 201706081626 201706081707
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

import geo_split_feasibility as geo
from week1_checks import MAX_CAN_GAP, RATIO_OK, read, resample

WRAP, SKIP, MOVING = 128.0, 6, 3.0
BINS = (5.0, 15.0)


def session(raw, sid, ratio):
    png = glob.glob(os.path.join(raw, "release_2019_07_25", "*", sid, "camera", "center", "png_timestamp.csv"))[0]
    d = pd.read_csv(png, skiprows=1, header=None, usecols=[0, 3]).to_numpy()[SKIP:]
    t, clk = d[:, 0], d[:, 1]
    clk_u = clk + WRAP * np.concatenate([[0], np.cumsum(np.diff(clk) < -WRAP / 2)])
    b, a = np.polyfit(clk_u, t, 1)
    resid = np.abs(t - (a + b * clk_u))
    csv = glob.glob(os.path.join(raw, "release_2019_07_08", "*", sid, "general", "csv"))[0]
    st = read(os.path.join(csv, "steer.csv"), ["t", "iso", "angle", "speed"])
    ve = read(os.path.join(csv, "vel.csv"), ["t", "iso", "v"])
    steer = resample(st["t"].to_numpy(float), st["angle"].to_numpy(float), t, MAX_CAN_GAP)
    v = resample(ve["t"].to_numpy(float), ve["v"].to_numpy(float), t, MAX_CAN_GAP) / ratio
    target = np.isfinite(steer) & np.isfinite(v) & (v > MOVING)
    return t, resid, np.abs(steer), target


def lost_mask(t, flag, window):
    """flagged, or a flagged frame within the previous `window` seconds"""
    c = np.concatenate([[0], np.cumsum(flag)])
    j0 = np.searchsorted(t, t - window, side="left")
    return (c[np.arange(len(t)) + 1] - c[j0]) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--checks", required=True, help="week1_checks output dir (sessions.csv, tracks_1hz.npz)")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.2, 1.0])
    ap.add_argument("--windows", type=float, nargs="+", default=[0.0, 5.0])
    a = ap.parse_args()

    ids, P, M, V = geo.load(os.path.join(a.checks, "tracks_1hz.npz"), set(a.exclude))
    test = geo.greedy_holdout(ids, P, M, 50.0, 25.0, [0.25])[0.25][0][1]
    val = geo.greedy_holdout(ids, P, M, 50.0, 25.0, [0.12], eligible=set(ids) - set(test))[0.12][0][1]
    split = {s: "test" for s in test} | {s: "val" for s in val} | {s: "train" for s in ids if s not in test and s not in val}

    sess = pd.read_csv(os.path.join(a.checks, "sessions.csv"), dtype={"session": str})
    rows = []
    for _, r in sess.iterrows():
        ratio = r.speed_ratio_can_over_gps
        ratio = ratio if np.isfinite(ratio) and RATIO_OK[0] < ratio < RATIO_OK[1] else 3.6
        rows.append((r.session, split.get(r.session, "none"), *session(a.raw, r.session, ratio)))

    tot = sum(int(tg.sum()) for *_, tg in rows)
    curve_all = sum(int((tg & (st >= BINS[1])).sum()) for *_, st, tg in rows) / tot
    print(f"prediction targets: {tot} frames; >= 15 deg among them: {curve_all:.3f}")
    base = {k: sum(int(tg.sum()) for _, sp, _, _, _, tg in rows if sp == k) for k in ("train", "val", "test", "none")}
    print("targets by split (design A):", base)
    for thr in a.thresholds:
        for W in a.windows:
            lost_by = {k: 0 for k in base}
            lost_curve = lost = sessions = 0
            for _, sp, t, res, st, tg in rows:
                m = lost_mask(t, res > thr, W) & tg
                n = int(m.sum())
                lost += n
                lost_by[sp] += n
                lost_curve += int((m & (st >= BINS[1])).sum())
                sessions += n > 0
            share = {k: round(lost_by[k] / base[k], 4) if base[k] else None for k in base}
            print(f"thr {thr:4.2f} s, window {W:3.0f} s: lost {lost} targets ({lost / tot:.4f}), sessions {sessions}; "
                  f">= 15 deg among lost {lost_curve / lost if lost else float('nan'):.3f}; lost share by split {share}")


if __name__ == "__main__":
    main()
