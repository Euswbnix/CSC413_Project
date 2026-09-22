"""Are frames and CAN really aligned? Cross-correlates image motion with yaw rate. Aggregates only.

extract_features.py already wrote, per frame, the horizontal image shift against the previous
frame. A right turn moves the image left, so that shift should track the CAN yaw rate times the
frame interval. This script estimates, per window, the lag that best aligns the two, using the
ROS timestamps as the common timeline.

  * a lag near zero everywhere means the frame timestamps and CAN agree;
  * a lag that tracks (ROS time - camera clock) means the ROS stamps are late, so those frames
    are mislabelled (see docs/hdd_week1_checks_2026-09-21.md section 1);
  * a lag that stays near zero while that residual grows means the residual is a clock artefact
    and labels are fine.

    python hdd/sync_check.py --features ~/data/hdd/features/dinov2_s10 --raw ~/data/hdd/raw \
        --out ~/data/hdd/checks/sync_check.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

MAX_CAN_GAP = 0.05
WINDOW_S = 60.0
MIN_MOVING = 3.0
MIN_YAW_STD = 2.0          # deg/s; a window with no turning cannot date itself
MAX_LAG_S = 2.0


def refuse_inside_git(path):
    d = os.path.abspath(os.path.dirname(path) or ".")
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            sys.exit(f"refusing to write HDD-derived output inside a git working tree ({d})")
        parent = os.path.dirname(d)
        if parent == d:
            return
        d = parent


def resample(t_src, v_src, t_dst, max_gap=MAX_CAN_GAP):
    ok = np.isfinite(v_src)
    ts, vs = t_src[ok], v_src[ok]
    out = np.interp(t_dst, ts, vs, left=np.nan, right=np.nan)
    i = np.clip(np.searchsorted(ts, t_dst), 1, ts.size - 1)
    out[(ts[i] - ts[i - 1]) > max_gap] = np.nan
    return out


def yaw_at(raw, session, t):
    csv = glob.glob(os.path.join(raw, "release_2019_07_08", "*", session, "general", "csv", "yaw.csv"))[0]
    d = pd.read_csv(csv, skiprows=1, header=None, usecols=[0, 2])
    return resample(d[0].to_numpy(float), d[2].to_numpy(float), t)


def detrend(x):
    return x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))


def lag_of(a, b, max_lag):
    """Lag in samples that best aligns b onto a, refined to sub-sample by a parabolic fit, with
    the correlation there. Positive means a happens LATER than b: np.correlate puts the peak at
    +k when a[n + k] ~ b[n]. Here a is the image motion and b is what CAN says, so a positive lag
    means the frame timestamps are later than the picture they carry."""
    a = (a - a.mean()) / (a.std() or 1)
    b = (b - b.mean()) / (b.std() or 1)
    c = np.correlate(a, b, mode="full") / len(a)
    mid = len(b) - 1
    seg = c[mid - max_lag:mid + max_lag + 1]
    k = int(np.argmax(np.abs(seg)))          # the relation may be negative: a right turn moves
    off = 0.0                                # the image left, so keep the sign and use |corr|
    if 0 < k < len(seg) - 1:                 # parabolic refinement around the peak
        y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den else 0.0
        off = float(np.clip(off, -1, 1))
    return k - max_lag + off, float(seg[k])


def session_windows(npz, raw, session):
    z = np.load(npz)
    t, t_cam, shift, speed = z["t_ros"], z["t_cam"], z["shift_px"], z["speed_mps"]
    # the shift at frame i covers the interval (i-1, i), so take the yaw rate at its midpoint
    dt = np.diff(t_cam, prepend=t_cam[0] - np.median(np.diff(t_cam)))
    yaw = yaw_at(raw, session, t - dt / 2)
    expected = yaw * dt                                  # degrees turned between frames
    resid = t - np.polyval(np.polyfit(t_cam, t, 1), t_cam)
    med_dt = float(np.median(np.diff(t)))
    max_lag = int(round(MAX_LAG_S / med_dt))
    rows = []
    n = int(WINDOW_S / med_dt)
    for s in range(0, len(t) - n, n):
        sl = slice(s, s + n)
        m = np.isfinite(expected[sl]) & np.isfinite(shift[sl]) & np.isfinite(speed[sl])
        if m.mean() < 0.9 or np.nanmean(speed[sl][m]) < MIN_MOVING:
            continue
        e, f = expected[sl][m], shift[sl][m]
        if e.std() < MIN_YAW_STD * med_dt or f.std() == 0 or len(e) < 4 * max_lag:
            continue
        # compare accumulated heading rather than per-frame motion: the per-frame shift is
        # quantised to whole profile columns (about 0.4 deg), which is the same size as a typical
        # frame-to-frame turn, and integrating averages that noise out
        e, f = detrend(np.cumsum(e)), detrend(np.cumsum(f))
        k, corr = lag_of(f, e, max_lag)
        rows.append(dict(t_mid=float(t[sl][m][len(e) // 2]), lag_s=float(k * med_dt), corr=corr,
                         resid_s=float(np.median(resid[sl])), n=int(m.sum())))
    return rows, float(np.median(np.abs(resid)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-corr", type=float, default=0.3, help="windows below this are unusable")
    a = ap.parse_args()
    refuse_inside_git(a.out)

    per_session, all_rows = {}, []
    for npz in sorted(glob.glob(os.path.join(a.features, "*.npz"))):
        session = os.path.basename(npz)[:-4]
        rows, med_resid = session_windows(npz, a.raw, session)
        good = [r for r in rows if abs(r["corr"]) >= a.min_corr]
        per_session[session] = dict(
            windows=len(rows), usable=len(good), median_resid_s=med_resid,
            median_lag_s=float(np.median([r["lag_s"] for r in good])) if good else None,
            iqr_lag_s=float(np.subtract(*np.percentile([r["lag_s"] for r in good], [75, 25]))) if good else None,
            median_corr=float(np.median([r["corr"] for r in good])) if good else None,
            corr_sign_positive=float(np.mean([r["corr"] > 0 for r in good])) if good else None,
            max_abs_resid_s=float(max((abs(r["resid_s"]) for r in rows), default=0.0)))
        all_rows += [dict(session=session, **r) for r in good]
        print(f"  {session}: {len(good)}/{len(rows)} usable windows, "
              f"lag {per_session[session]['median_lag_s']}, corr {per_session[session]['median_corr']}", flush=True)

    lags = np.array([r["lag_s"] for r in all_rows])
    resids = np.array([r["resid_s"] for r in all_rows])
    summary = dict(sessions=len(per_session), usable_windows=len(all_rows),
                   lag_p10_p50_p90=np.percentile(lags, [10, 50, 90]).round(4).tolist() if len(lags) else None,
                   share_abs_lag_over_100ms=float(np.mean(np.abs(lags) > 0.1)) if len(lags) else None,
                   corr_p10_p50=np.percentile([abs(r["corr"]) for r in all_rows], [10, 50]).round(3).tolist() if all_rows else None,
                   share_corr_positive=float(np.mean([r["corr"] > 0 for r in all_rows])) if all_rows else None)
    big = np.abs(resids) > 0.2
    if big.sum() > 5:
        # does the lag follow the ROS-minus-clock residual? slope 1 = the ROS stamps are late
        slope, intercept = np.polyfit(resids[big], lags[big], 1)
        summary["drift_windows"] = int(big.sum())
        summary["lag_vs_resid_slope"] = round(float(slope), 3)
        summary["lag_vs_resid_intercept_s"] = round(float(intercept), 4)
        summary["median_lag_in_drift_windows_s"] = round(float(np.median(lags[big])), 4)
        summary["median_lag_elsewhere_s"] = round(float(np.median(lags[~big])), 4)
    json.dump(dict(summary=summary, sessions=per_session, windows=all_rows), open(a.out, "w"), indent=1)
    os.chmod(a.out, 0o600)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
