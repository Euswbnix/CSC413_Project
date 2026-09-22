"""Freeze the HDD split (design A) and everything the analysis needs to describe it.

Design A, as decided on 2026-09-22 (docs/discussion_2026-09-21_next_steps.md section 1):
whole sessions held out in route groups, found by a deterministic greedy search on the 1 Hz GPS
tracks alone -- no labels, no model outputs. Then the sync-check acceptance rule: a val or test
session that is not verified (>= 10 usable windows and |lag| <= 100 ms) is dropped from the study
entirely; sessions already in train stay in train whatever their evidence.

Writes to <out> (server only, owner-readable):
  split.json        per session: split, route cluster, hours, sync evidence; plus the parameters,
                    the seed-free search settings and per-split totals
  frame_distance.npz  per val/test session: distance from every recorded frame to the nearest
                    training track point, in metres (for the low-overlap reporting D2 requires)
and prints the aggregate table plus the SHA-256 of split.json, which is what goes in the repo.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geo_split_feasibility as geo

EARTH_R = 6_371_000.0
MIN_WINDOWS = 10
MAX_LAG_S = 0.1


def refuse_inside_git(path):
    d = os.path.abspath(path if os.path.isdir(path) else os.path.dirname(path) or ".")
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            sys.exit(f"refusing to write HDD-derived output inside a git working tree ({d})")
        parent = os.path.dirname(d)
        if parent == d:
            return
        d = parent


def verified(sync, session):
    v = sync.get(session, {})
    lag = v.get("median_lag_s")
    return v.get("usable", 0) >= MIN_WINDOWS and lag is not None and abs(lag) <= MAX_LAG_S


def frame_distances(raw, features, session, tree, lat0, lon0):
    """Distance from each recorded frame of `session` to the nearest training track point."""
    z = np.load(os.path.join(features, f"{session}.npz"))
    t = z["t_ros"]
    pos = glob.glob(os.path.join(raw, "release_2019_07_08", "*", session, "general", "csv", "rtk_pos.csv"))
    if not pos:
        return None
    d = pd.read_csv(pos[0], skiprows=1, header=None, usecols=[0, 2, 3])
    tp, a, b = (d[c].to_numpy(float) for c in d.columns)
    lat, lon = (a, b) if np.nanmedian(np.abs(a)) <= 90 else (b, a)
    ok = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) > 1)
    if ok.sum() < 100:
        return None
    x = np.interp(t, tp[ok], np.radians(lon[ok] - lon0) * EARTH_R * np.cos(np.radians(lat0)))
    y = np.interp(t, tp[ok], np.radians(lat[ok] - lat0) * EARTH_R)
    dist, _ = tree.query(np.column_stack([x, y]))
    return dist.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--sync", required=True, help="sync_check.json from --signal steer")
    ap.add_argument("--features", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="sessions with unusable GPS")
    ap.add_argument("--drop", nargs="*", default=[], help="sessions excluded from the study entirely")
    ap.add_argument("--radius", type=float, default=50.0)
    ap.add_argument("--cell", type=float, default=25.0)
    ap.add_argument("--test-share", type=float, default=0.25)
    ap.add_argument("--val-share", type=float, default=0.12)
    ap.add_argument("--cluster-threshold", type=float, default=0.5)
    a = ap.parse_args()
    refuse_inside_git(a.out)
    os.makedirs(a.out, exist_ok=True)
    os.chmod(a.out, 0o700)

    ids, P, M, V = geo.load(a.tracks, set(a.exclude) | set(a.drop))
    test = geo.greedy_holdout(ids, P, M, a.radius, a.cell, [a.test_share])[a.test_share][0][1]
    val = geo.greedy_holdout(ids, P, M, a.radius, a.cell, [a.val_share],
                             eligible=set(ids) - set(test))[a.val_share][0][1]
    sync = json.load(open(a.sync))["sessions"]

    split = {}
    for s in ids:
        if s in test or s in val:
            split[s] = (s in test and "test" or "val") if verified(sync, s) else "dropped"
        else:
            split[s] = "train"
    comps, _ = geo.route_clusters(ids, P, M, a.radius, a.cell, a.cluster_threshold)
    cluster = {ids[i]: c for c, comp in enumerate(comps) for i in comp}

    hours = {s: float(M[s].sum()) / 3600 for s in ids}
    keep = [s for s in ids if split[s] == "train"]
    lat0 = float(np.median(np.concatenate([np.load(a.tracks)[s][:, 0] for s in keep])))
    lon0 = float(np.median(np.concatenate([np.load(a.tracks)[s][:, 1] for s in keep])))
    tree = cKDTree(np.vstack([P[s] for s in keep]))

    dists = {}
    for s in ids:
        if split[s] in ("val", "test"):
            d = frame_distances(a.raw, a.features, s, tree, lat0, lon0)
            if d is not None:
                dists[s] = d
    np.savez_compressed(os.path.join(a.out, "frame_distance.npz"), **dists)

    sessions = {s: dict(split=split[s], cluster=int(cluster[s]), moving_hours=round(hours[s], 3),
                        sync_windows=int(sync.get(s, {}).get("usable", 0)),
                        sync_lag_s=sync.get(s, {}).get("median_lag_s")) for s in ids}
    totals = {}
    for k in ("train", "val", "test", "dropped"):
        g = [s for s in ids if split[s] == k]
        cl = {cluster[s] for s in g}
        cw = np.array([sum(hours[s] for s in g if cluster[s] == c) for c in cl]) if cl else np.zeros(1)
        totals[k] = dict(sessions=len(g), hours=round(sum(hours[s] for s in g), 2),
                         clusters=len(cl), kish_clusters=round(geo.kish(cw), 1))
    out = dict(design="A", decided="2026-09-22", parameters=dict(
        radius_m=a.radius, raster_cell_m=a.cell, test_share=a.test_share, val_share=a.val_share,
        cluster_threshold=a.cluster_threshold, min_sync_windows=MIN_WINDOWS, max_sync_lag_s=MAX_LAG_S,
        excluded_gps=sorted(a.exclude), dropped_before_split=sorted(a.drop),
        tracks=os.path.basename(a.tracks), sync=os.path.basename(a.sync)),
        totals=totals, sessions=sessions)
    path = os.path.join(a.out, "split.json")
    open(path, "w").write(json.dumps(out, indent=1) + "\n")
    for f in ("split.json", "frame_distance.npz"):
        os.chmod(os.path.join(a.out, f), 0o600)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()

    print(f"{'split':>8} {'sessions':>9} {'hours':>7} {'clusters':>9} {'Kish':>6}")
    for k, v in totals.items():
        print(f"{k:>8} {v['sessions']:9d} {v['hours']:7.2f} {v['clusters']:9d} {v['kish_clusters']:6.1f}")
    for k in ("val", "test"):
        d = np.concatenate([dists[s] for s in ids if split[s] == k and s in dists])
        print(f"  {k}: frames beyond 200 m of any training track: {np.mean(d > 200):.1%}, beyond 50 m: {np.mean(d > 50):.1%}")
    print(f"split.json sha256 {digest}")


if __name__ == "__main__":
    main()
