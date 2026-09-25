"""Do HDD frames and CAN steering share a timeline? Cross-correlates image motion with yaw rate.

extract_features.py records, per frame, the horizontal image shift against the previous frame. A
turn moves the image sideways, so that shift should track the CAN yaw rate times the frame
interval. Per 60 s window this reports the lag that best aligns the two on the ROS timeline.

The default compares against the steering angle, which is what labels the frames. The CAN yaw
rate trails the picture by about 0.1 s (gyro filtering), so timing labels against it invents a
correction that is not there.

Sign: lag > 0 means the image signal happens LATER than CAN says. A negative lag means the frame
timestamps are later than the picture they carry, i.e. the camera path adds delay before the
timestamp is taken.

Two things this script had wrong before 2026-09-22, both found by injecting known lags into real
windows (--selftest, and hdd/tests/test_sync_estimator.py):

  * correlating the ACCUMULATED heading pins the estimate at zero for small lags: integration
    concentrates the signal near DC, where a lag barely changes the shape;
  * dividing np.correlate by the full length instead of the overlap adds a triangular taper that
    pulls the peak toward zero as well.

So the estimate uses the per-frame signals and a Pearson correlation computed on the overlapping
samples only. Run --selftest after touching any of it.

    python hdd/sync_check.py --features ~/workspace/hdd/features/dinov2_s10 --raw ~/workspace/hdd/raw \
        --out ~/workspace/hdd/checks/sync_check.json --signal steer --max-lag-s 5
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
MIN_OVERLAP = 30


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


def ncc_lag(a, b, max_lag):
    """Lag k maximising |corr(a[n+k], b[n])|, computed on the overlap only, with the correlation
    there and the curvature of the peak. Returns (lag, corr, peak_sharpness)."""
    rs = np.full(2 * max_lag + 1, np.nan)
    for j, k in enumerate(range(-max_lag, max_lag + 1)):
        x, y = (a[k:], b[:len(b) - k or None]) if k >= 0 else (a[:k], b[-k:])
        n = min(len(x), len(y))
        if n < MIN_OVERLAP:
            continue
        x, y = x[:n], y[:n]
        if x.std() == 0 or y.std() == 0:
            continue
        rs[j] = np.corrcoef(x, y)[0, 1]
    if np.all(np.isnan(rs)):
        return None, None, None
    j = int(np.nanargmax(np.abs(rs)))
    r = float(rs[j])
    # how much better the peak is than the rest: a flat profile means the lag is not identified
    others = np.abs(rs[max(0, j - 3):j + 4])
    sharp = float(abs(r) - np.nanmedian(np.abs(np.delete(rs, slice(max(0, j - 3), j + 4)))))
    return j - max_lag, r, sharp


def session_windows(npz, raw, session, max_lag_s, signal="yaw"):
    """`signal` picks what the image motion is compared against: the CAN yaw rate, or the turn
    rate implied by the steering angle (steer x speed, up to a constant). The two CAN channels
    need not share a latency, and it is the steering one that labels the frames."""
    z = np.load(npz)
    t, t_cam, shift, speed = z["t_ros"], z["t_cam"], z["shift_px"], z["speed_mps"]
    dt = np.diff(t_cam, prepend=t_cam[0] - np.median(np.diff(t_cam)))
    if signal == "yaw":
        rate = yaw_at(raw, session, t - dt / 2)     # the shift covers (i-1, i): take its midpoint
    else:
        rate = z["steer"] * speed                    # proportional to the yaw rate it produces
    expected = rate * dt                             # degrees turned between the two frames
    resid = t - np.polyval(np.polyfit(t_cam, t, 1), t_cam)
    med_dt = float(np.median(np.diff(t)))
    max_lag = int(round(max_lag_s / med_dt))
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
        k, corr, sharp = ncc_lag(f, e, max_lag)
        if k is None:
            continue
        rows.append(dict(t_mid=float(t[sl][m][len(e) // 2]), lag_s=float(k * med_dt), corr=corr,
                         sharpness=sharp, resid_s=float(np.median(resid[sl])), n=int(m.sum())))
    return rows, float(np.median(np.abs(resid))), float(np.max(np.abs(resid)))


def selftest(features, raw, injections=(0.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5), n_sessions=6, max_lag_s=2.0):
    """Recover known lags injected into real windows. The estimate must move one-for-one."""
    wins = []
    for npz in sorted(glob.glob(os.path.join(features, "*.npz")))[:n_sessions]:
        z = np.load(npz)
        t, t_cam, shift, speed = z["t_ros"], z["t_cam"], z["shift_px"], z["speed_mps"]
        dt = np.diff(t_cam, prepend=t_cam[0] - np.median(np.diff(t_cam)))
        e_all = yaw_at(raw, os.path.basename(npz)[:-4], t - dt / 2) * dt
        med_dt = float(np.median(np.diff(t)))
        n = int(WINDOW_S / med_dt)
        for s in range(0, len(t) - n, n):
            sl = slice(s, s + n)
            m = np.isfinite(e_all[sl]) & np.isfinite(shift[sl]) & np.isfinite(speed[sl])
            if m.mean() < 0.9 or np.nanmean(speed[sl][m]) < MIN_MOVING:
                continue
            e, f = e_all[sl][m], shift[sl][m]
            if e.std() < MIN_YAW_STD * med_dt or f.std() == 0:
                continue
            wins.append((e, f, med_dt))
            if len(wins) >= 60:
                break
    print(f"self-test on {len(wins)} real windows; the estimate should move one-for-one")
    base = None
    out = []
    for inj in injections:
        got = []
        for e, f, med_dt in wins:
            k = int(round(inj / med_dt))
            max_lag = int(round(max_lag_s / med_dt))
            fs = np.roll(f, k)
            cut = abs(k) + 1
            fs, ee = (fs[cut:-cut], e[cut:-cut]) if cut > 1 else (fs, e)
            if len(ee) < 4 * max_lag:
                continue
            kk, r, _ = ncc_lag(fs, ee, max_lag)
            if kk is not None:
                got.append(kk * med_dt)
        med = float(np.median(got))
        base = med if inj == 0.0 and base is None else base
        out.append((inj, med, len(got)))
        print(f"  injected {inj:+.2f} s -> estimated {med:+.3f} s  ({len(got)} windows)"
              + (f", minus the no-injection estimate: {med - base:+.3f} s" if base is not None else ""))
    slope = np.polyfit([o[0] for o in out], [o[1] for o in out], 1)[0]
    print(f"  slope {slope:.3f} (1.0 = unbiased), estimate without injection {base:+.3f} s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out")
    ap.add_argument("--max-lag-s", type=float, default=5.0)
    ap.add_argument("--min-corr", type=float, default=0.3)
    ap.add_argument("--min-sharpness", type=float, default=0.1,
                    help="peak must beat the rest of the lag profile by this much")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sessions", nargs="*", help="only these sessions")
    ap.add_argument("--signal", choices=("steer", "yaw"), default="steer",
                    help="compare the image motion against the CAN yaw rate or the steering angle")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.features, a.raw, max_lag_s=a.max_lag_s)
        return
    if not a.out:
        sys.exit("--out is required")
    refuse_inside_git(a.out)

    per_session, all_rows = {}, []
    for npz in sorted(glob.glob(os.path.join(a.features, "*.npz"))):
        session = os.path.basename(npz)[:-4]
        if a.sessions and session not in a.sessions:
            continue
        rows, med_resid, max_resid = session_windows(npz, a.raw, session, a.max_lag_s, a.signal)
        good = [r for r in rows if abs(r["corr"]) >= a.min_corr and r["sharpness"] >= a.min_sharpness]
        lags = [r["lag_s"] for r in good]
        per_session[session] = dict(
            windows=len(rows), usable=len(good), median_resid_s=med_resid, max_abs_resid_s=max_resid,
            median_lag_s=float(np.median(lags)) if good else None,
            iqr_lag_s=float(np.subtract(*np.percentile(lags, [75, 25]))) if good else None,
            median_corr=float(np.median([abs(r["corr"]) for r in good])) if good else None)
        all_rows += [dict(session=session, **r) for r in good]

    lags = np.array([r["lag_s"] for r in all_rows])
    resids = np.array([r["resid_s"] for r in all_rows])
    med = np.array([v["median_lag_s"] for v in per_session.values() if v["median_lag_s"] is not None])
    summary = dict(signal=a.signal, sessions=len(per_session), sessions_with_windows=len(med), usable_windows=len(all_rows),
                   window_lag_p10_p50_p90=np.percentile(lags, [10, 50, 90]).round(4).tolist() if len(lags) else None,
                   session_lag_p10_p50_p90=np.percentile(med, [10, 50, 90]).round(4).tolist() if len(med) else None,
                   corr_p10_p50=np.percentile([abs(r["corr"]) for r in all_rows], [10, 50]).round(3).tolist() if all_rows else None,
                   sessions_without_windows=[k for k, v in per_session.items() if v["usable"] == 0])
    big = np.abs(resids) > 0.2
    if big.sum() > 5 and len(set(np.round(resids[big], 3))) > 3:
        slope, intercept = np.polyfit(resids[big], lags[big], 1)
        sess = {r["session"] for r, b in zip(all_rows, big) if b}
        summary["drift"] = dict(windows=int(big.sum()), sessions=len(sess), slope=round(float(slope), 3),
                                intercept_s=round(float(intercept), 4),
                                median_lag_drift_s=round(float(np.median(lags[big])), 4),
                                median_lag_rest_s=round(float(np.median(lags[~big])), 4))
    json.dump(dict(summary=summary, sessions=per_session, windows=all_rows), open(a.out, "w"), indent=1)
    os.chmod(a.out, 0o600)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
