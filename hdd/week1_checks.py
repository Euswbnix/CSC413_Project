"""Week-1 checks on the Honda HDD release, from CAN/GPS/timestamp CSVs only (no frames).

Per session:
  * frame timing: png_timestamp.csv spacing, the slow start-up frames, gaps
  * CAN timing: steer.csv spacing and gaps
  * speed units: CAN speed vs speed derived from RTK positions
  * sign conventions: CAN yaw rate vs GPS heading rate, steering angle vs yaw rate
  * steering-to-curvature slope (steering ratio x wheelbase), label at frame times
  * steering distribution at frame times, in the |y| bins used by metrics.py

Dataset level: totals, and how far the TRN test sessions' GPS tracks are from any training
track (the published split is supposed to be geographic).

Writes <out>/sessions.csv and <out>/summary.json. Usage:
    python hdd/week1_checks.py --raw ~/data/hdd/raw --split trn_data_info.json --out ~/data/hdd/checks
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

BINS = (5.0, 15.0)          # |steer| bin edges in degrees, same as metrics.BIN_EDGES
MOVING = 3.0                # m/s; below this, curvature and heading are noise
EARTH_R = 6_371_000.0


FALLBACK_RATIO = float("nan")   # CAN speed units per m/s, for sessions without GPS


def read(path, cols):
    """Read a headerless-after-'#' CSV, skipping the string columns (named iso/path)."""
    keep = [i for i, c in enumerate(cols) if c not in ("iso", "path")]
    df = pd.read_csv(path, skiprows=1, header=None, usecols=keep, engine="c")
    df.columns = [cols[i] for i in keep]
    return df


def init_worker(fallback_ratio):
    global FALLBACK_RATIO
    FALLBACK_RATIO = fallback_ratio


def latlon(df):
    """The rtk_pos header says lng,lat but the values are lat,lng; decide from the ranges."""
    a, b = df["c1"].to_numpy(), df["c2"].to_numpy()
    if np.nanmedian(np.abs(a)) <= 90 and np.nanmedian(np.abs(b)) > 90:
        return a, b, True
    return b, a, False


def enu(lat, lon, lat0, lon0):
    x = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return x, y


def dt_stats(t):
    d = np.diff(t)
    if d.size == 0:
        return {}
    med = float(np.median(d))
    return {
        "n": int(t.size), "span_s": float(t[-1] - t[0]), "median_dt": med,
        "p01_dt": float(np.percentile(d, 1)), "p99_dt": float(np.percentile(d, 99)),
        "max_dt": float(d.max()), "n_nonpos": int((d <= 0).sum()),
        "n_gap_1p5x": int((d > 1.5 * med).sum()), "gap_time_s": float(d[d > 1.5 * med].sum()),
    }


def resample(t_src, v_src, t_dst):
    ok = np.isfinite(v_src)
    return np.interp(t_dst, t_src[ok], v_src[ok], left=np.nan, right=np.nan)


def sign_agreement(a, b, mask):
    m = mask & (a != 0) & (b != 0)
    return float((np.sign(a[m]) == np.sign(b[m])).mean()) if m.sum() else float("nan"), int(m.sum())


def session(args):
    sdir, png_csv = args
    sid = os.path.basename(sdir)
    csv = os.path.join(sdir, "general", "csv")
    out = {"session": sid, "day": os.path.basename(os.path.dirname(sdir))}

    frames = read(png_csv, ["t", "iso", "path", "cam_clock", "cam_counter"])
    tf = frames["t"].to_numpy(float)
    out["frames"] = dt_stats(tf)
    d = np.diff(tf)
    # the first few frames arrive about 1 s apart before the stream reaches frame rate
    first_fast = int(np.argmax(d < 0.1)) if (d < 0.1).any() else len(d)
    out["frames"]["slow_startup_frames"] = first_fast
    cc = np.diff(frames["cam_counter"].to_numpy())
    out["frames"]["counter_steps_not_1"] = int((cc != 1).sum())
    tf_run = tf[first_fast:]
    out["frames_steady"] = dt_stats(tf_run)

    steer = read(os.path.join(csv, "steer.csv"), ["t", "iso", "angle", "speed"])
    yaw = read(os.path.join(csv, "yaw.csv"), ["t", "iso", "yaw"])
    vel = read(os.path.join(csv, "vel.csv"), ["t", "iso", "v"])
    ts = steer["t"].to_numpy(float)
    out["steer_can"] = dt_stats(ts)

    # label at frame times: nearest-sample distance tells how tight the sync can be
    idx = np.clip(np.searchsorted(ts, tf_run), 1, len(ts) - 1)
    near = np.minimum(np.abs(ts[idx] - tf_run), np.abs(ts[idx - 1] - tf_run))
    out["label_sync_ms"] = {"median": float(np.median(near) * 1e3), "p99": float(np.percentile(near, 99) * 1e3),
                            "max": float(near.max() * 1e3),
                            "frames_outside_can": int(((tf_run < ts[0]) | (tf_run > ts[-1])).sum())}

    # everything on a common 10 Hz grid for the physics checks
    t0, t1 = max(ts[0], yaw["t"].iloc[0], vel["t"].iloc[0]), min(ts[-1], yaw["t"].iloc[-1], vel["t"].iloc[-1])
    g = np.arange(t0, t1, 0.1)
    st = resample(ts, steer["angle"].to_numpy(float), g)
    yw = resample(yaw["t"].to_numpy(float), yaw["yaw"].to_numpy(float), g)
    v_can = resample(vel["t"].to_numpy(float), vel["v"].to_numpy(float), g)

    pos_path = os.path.join(csv, "rtk_pos.csv")
    out["has_gps"] = os.path.exists(pos_path)
    v_gps = hd_rate = None
    if out["has_gps"]:
        pos = read(pos_path, ["t", "iso", "c1", "c2"])
        lat, lon, swapped = latlon(pos)
        out["gps_header_swapped"] = swapped
        ok = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) > 1)
        tp = pos["t"].to_numpy(float)[ok]
        lat, lon = lat[ok], lon[ok]
        out["gps_fix_fraction"] = float(ok.mean())
        if tp.size > 100:
            x, y = enu(lat, lon, lat[0], lon[0])
            g1 = np.arange(max(t0, tp[0]), min(t1, tp[-1]), 1.0)
            xs, ys = np.interp(g1, tp, x), np.interp(g1, tp, y)
            v1 = np.hypot(np.diff(xs), np.diff(ys))
            vc = np.interp(g1[:-1] + 0.5, g, v_can)
            m = (v1 > MOVING) & np.isfinite(vc) & (vc > 0)
            out["speed_ratio_can_over_gps"] = float(np.median(vc[m] / v1[m])) if m.sum() > 30 else float("nan")
            out["track_1hz"] = {"lat": np.interp(g1, tp, lat).round(6).tolist(),
                                "lon": np.interp(g1, tp, lon).round(6).tolist()}
            out["lat0"], out["lon0"] = float(np.median(lat)), float(np.median(lon))
            # heading rate from the track itself (clockwise from north), independent of any
            # rtk_track column whose meaning is undocumented
            hdg = np.unwrap(np.arctan2(np.gradient(xs), np.gradient(ys)))
            hd_rate = {"pos": np.degrees(np.gradient(hdg))}
            g1_moving = np.hypot(np.gradient(xs), np.gradient(ys)) > MOVING
        trk_path = os.path.join(csv, "rtk_track.csv")
        if os.path.exists(trk_path) and hd_rate is not None:
            trk = read(trk_path, ["t", "iso", "course", "heading", "pitch", "roll"])
            for col in ("course", "heading"):
                h = np.unwrap(np.radians(trk[col].to_numpy(float)))
                hd_rate[col] = np.degrees(np.gradient(resample(trk["t"].to_numpy(float), h, g1)))
    out["has_gps_track"] = hd_rate is not None

    ratio = out.get("speed_ratio_can_over_gps", float("nan"))
    out["speed_ratio_source"] = "gps" if np.isfinite(ratio) and ratio > 0 else "fallback"
    if out["speed_ratio_source"] == "fallback":
        ratio = FALLBACK_RATIO           # nan leaves every speed-gated statistic empty
    v_ms = v_can / ratio
    moving = np.isfinite(v_ms) & (v_ms > MOVING)
    turning = moving & np.isfinite(yw) & (np.abs(yw) > 3.0)
    out["n_moving_10hz"] = int(moving.sum())
    out["sign_steer_vs_yaw"], out["n_sign_steer_yaw"] = sign_agreement(st, yw, turning & np.isfinite(st))
    if hd_rate is not None:
        # 1 Hz comparison: CAN yaw rate vs heading rate; agreement near 0 means opposite conventions
        yw1 = np.interp(g1, g, np.nan_to_num(yw))
        for name, hr in hd_rate.items():
            m = g1_moving & np.isfinite(hr) & (np.abs(yw1) > 3.0)
            out[f"sign_{name}_rate_vs_yaw"], _ = sign_agreement(hr, yw1, m)
            out[f"corr_{name}_rate_vs_yaw"] = float(np.corrcoef(hr[m], yw1[m])[0, 1]) if m.sum() > 30 else float("nan")
    # curvature kappa = yaw_rate / v (1/m); steering-wheel deg ~= slope * kappa at low lateral accel
    kappa = np.radians(yw) / np.where(v_ms > 0, v_ms, np.nan)
    m = moving & np.isfinite(kappa) & np.isfinite(st) & (np.abs(kappa) < 0.2)
    if m.sum() > 100:
        out["steer_deg_per_kappa"] = float(np.sum(st[m] * kappa[m]) / np.sum(kappa[m] ** 2))
        out["corr_steer_vs_kappa"] = float(np.corrcoef(st[m], kappa[m])[0, 1])

    # steering distribution at steady frame times, split by moving / stopped
    sf = resample(ts, steer["angle"].to_numpy(float), tf_run)
    vf = resample(g, v_ms, tf_run)
    a = np.abs(sf[np.isfinite(sf)])
    mv = np.isfinite(sf) & np.isfinite(vf) & (vf > MOVING)
    am = np.abs(sf[mv])
    out["steer_frames"] = {
        "n": int(a.size), "n_moving": int(am.size),
        "bins_all": [float((a < BINS[0]).mean()), float(((a >= BINS[0]) & (a < BINS[1])).mean()), float((a >= BINS[1]).mean())],
        "bins_moving": [float((am < BINS[0]).mean()), float(((am >= BINS[0]) & (am < BINS[1])).mean()), float((am >= BINS[1]).mean())] if am.size else None,
        "abs_pcts_moving": np.percentile(am, [50, 90, 99]).round(2).tolist() if am.size else None,
        "mean": float(np.mean(sf[np.isfinite(sf)])),
        "hist_moving": np.histogram(np.clip(sf[mv], -540, 540), bins=np.arange(-540, 541, 5))[0].tolist(),
    }
    return out


def geo_overlap(rows, split):
    from scipy.spatial import cKDTree
    have = {r["session"]: r for r in rows if "track_1hz" in r}
    lat0 = np.median([r["lat0"] for r in have.values()])
    lon0 = np.median([r["lon0"] for r in have.values()])

    def pts(ids):
        xy = [np.column_stack(enu(np.array(have[s]["track_1hz"]["lat"]), np.array(have[s]["track_1hz"]["lon"]), lat0, lon0))
              for s in ids if s in have]
        return np.vstack(xy) if xy else np.zeros((0, 2))

    train, test = split["train_session_set"], split["test_session_set"]
    tree = cKDTree(pts(train))
    res = {"train_with_gps": sum(s in have for s in train), "test_with_gps": sum(s in have for s in test)}
    per = {}
    for s in test:
        if s not in have:
            continue
        d, _ = tree.query(pts([s]))
        per[s] = {r: float((d < r).mean()) for r in (50, 200, 1000)}
    for r in (50, 200, 1000):
        res[f"test_points_within_{r}m_of_train"] = float(np.mean([p[r] for p in per.values()]))
    res["test_sessions_mostly_on_train_roads"] = int(sum(p[50] > 0.5 for p in per.values()))
    res["per_test_session_within_50m"] = {s: round(p[50], 3) for s, p in per.items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--split", required=True, help="TRN.pytorch data/data_info.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fallback-ratio", type=float, default=float("nan"),
                    help="CAN speed units per m/s for sessions without GPS (read it off a first run)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sessions = sorted(glob.glob(os.path.join(a.raw, "release_2019_07_08", "*", "[0-9]" * 12)))
    pngs = {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p)))): p
            for p in glob.glob(os.path.join(a.raw, "release_2019_07_25", "*", "*", "camera", "center", "png_timestamp.csv"))}
    jobs = [(s, pngs[os.path.basename(s)]) for s in sessions if os.path.basename(s) in pngs]
    missing = [os.path.basename(s) for s in sessions if os.path.basename(s) not in pngs]
    with ProcessPoolExecutor(a.workers, initializer=init_worker, initargs=(a.fallback_ratio,)) as ex:
        rows = list(ex.map(session, jobs))

    split = json.load(open(a.split))["HDD"]
    flat = pd.json_normalize([{k: v for k, v in r.items() if k != "track_1hz"} for r in rows], sep=".")
    flat = flat.drop(columns=[c for c in flat.columns if c.endswith("hist_moving")])
    flat.to_csv(os.path.join(a.out, "sessions.csv"), index=False)

    # 1 Hz GPS tracks for split design; server-only (the output directory is owner-only)
    np.savez_compressed(os.path.join(a.out, "tracks_1hz.npz"),
                        **{r["session"]: np.column_stack([r["track_1hz"]["lat"], r["track_1hz"]["lon"]])
                           for r in rows if "track_1hz" in r})
    hist = np.sum([r["steer_frames"]["hist_moving"] for r in rows], axis=0)
    ids = {r["session"] for r in rows}
    summary = {
        "sessions": len(rows), "sessions_without_frame_timestamps": missing,
        "hours_frames_steady": float(sum(r["frames_steady"]["span_s"] for r in rows) / 3600),
        "frames_steady_total": int(sum(r["frames_steady"]["n"] for r in rows)),
        "split": {"train": len(split["train_session_set"]), "test": len(split["test_session_set"]),
                  "in_release_not_in_split": sorted(ids - set(split["train_session_set"]) - set(split["test_session_set"])),
                  "in_split_not_in_release": sorted((set(split["train_session_set"]) | set(split["test_session_set"])) - ids)},
        "geo": geo_overlap(rows, split),
        "steer_hist_moving": {"edges_deg": [-540, 540, 5], "counts": hist.tolist()},
    }
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "steer_hist_moving"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
