"""Can HDD be split so that val/test roads are unseen in training? Prints aggregates only.

Reads tracks_1hz.npz written by week1_checks.py (1 Hz lat/lon per session, server-only) and,
optionally, TRN.pytorch's data_info.json. Nothing it prints can place a session on a map.

All shares are of *usable* 1 Hz points: moving (displacement > 3 m/s, the threshold week1_checks.py
uses for the steering statistics), with a steady-rate video frame within 0.5 s and a steering label
that does not bridge a CAN gap; over the sessions with usable GPS. The real split script works on
frame timestamps instead, so its shares will differ a little.

  A. session level: share of each session's points within R of any other session's track
  B. whole-session holdout: greedy search for a group of sessions sharing few roads with the rest
  C. the TRN split, pooled and per-session
  D. random block splits of side B (63/12/25% of blocks to train/val/test):
       val points within R of train are dropped; test points within R of train or val are dropped;
       a sample is kept only if its target point and its whole history window (W s) are in the
       same split and not dropped
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

EARTH_R = 6_371_000.0
MOVING = 3.0



def refuse_inside_git(path):
    """HDD-derived outputs must never land in a git working tree (licence §4.b; this repo is public)."""
    d = os.path.abspath(path if os.path.isdir(path) else os.path.dirname(path) or ".")
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            sys.exit(f"refusing to write HDD-derived output inside a git working tree ({d}); use ~/workspace/hdd/")
        parent = os.path.dirname(d)
        if parent == d:
            return
        d = parent

def origin(path, exclude):
    """The local frame every script must share: the median position over the loaded sessions.
    Anything that queries a tree built by load() has to convert its own points with this."""
    z = np.load(path)
    ids = sorted(s for s in z.files if s not in exclude)
    return (float(np.median(np.concatenate([z[s][:, 0] for s in ids]))),
            float(np.median(np.concatenate([z[s][:, 1] for s in ids]))))


def load(path, exclude):
    z = np.load(path)
    ids = sorted(s for s in z.files if s not in exclude)
    lat0, lon0 = origin(path, exclude)
    P, M, V = {}, {}, {}
    for s in ids:
        ll = z[s]
        xy = np.column_stack([np.radians(ll[:, 1] - lon0) * EARTH_R * np.cos(np.radians(lat0)),
                              np.radians(ll[:, 0] - lat0) * EARTH_R])
        P[s] = xy
        V[s] = np.hypot(*np.gradient(xy, axis=0).T)
        # usable = moving, with video and a valid steering label (third column from week1_checks);
        # overlap trees still use every point, which over- rather than under-states leakage
        M[s] = (V[s] > MOVING) & (ll[:, 2] > 0.5 if ll.shape[1] > 2 else True)
    return ids, P, M, V


def session_overlap(ids, P, M, r):
    allp = np.vstack([P[s] for s in ids])
    owner = np.concatenate([np.full(len(P[s]), i) for i, s in enumerate(ids)])
    tree = cKDTree(allp)
    out = []
    for i, s in enumerate(ids):
        hits = tree.query_ball_point(P[s][M[s]], r=r)
        out.append(np.mean([np.any(owner[h] != i) for h in hits]))
    return np.array(out)


def exact_group_overlap(ids, P, M, group, r):
    rest = cKDTree(np.vstack([P[s] for s in ids if s not in group]))
    q = np.vstack([P[s][M[s]] for s in group])
    d, _ = rest.query(q, distance_upper_bound=r)
    return float(np.isfinite(d).mean())


def greedy_holdout(ids, P, M, r, cell, targets, eligible=None):
    """Grow a held-out group one session at a time, always adding the session that keeps the
    group's overlap with the remaining sessions lowest; every eligible session is tried as the
    start. Sessions outside `eligible` can never join the group but still count as "the rest".
    Overlap here is on a raster: a point overlaps when a remaining session passes within
    ceil(r / cell) cells, which over-approximates distance r. The winners are re-scored exactly."""
    k = int(np.ceil(r / cell))
    off = np.array([(dx, dy) for dx in range(-k, k + 1) for dy in range(-k, k + 1)], dtype=np.int64)

    def key(c):
        return (c[:, 0] + (1 << 24)) * (1 << 25) + (c[:, 1] + (1 << 24))

    cov_k, pts_k = [], []
    for s in ids:
        c = np.floor(P[s] / cell).astype(np.int64)
        cu = np.unique(c, axis=0)
        cov_k.append(np.unique(key((cu[:, None, :] + off[None]).reshape(-1, 2))))
        pts_k.append(key(c[M[s]]))
    allk, inv = np.unique(np.concatenate(cov_k + pts_k), return_inverse=True)
    parts = np.split(inv.ravel(), np.cumsum([len(a) for a in cov_k + pts_k])[:-1])
    cov, pts = parts[: len(ids)], parts[len(ids):]
    w = np.array([len(p) for p in pts], dtype=float)
    total = w.sum()
    R0 = np.zeros(len(allk), dtype=np.int32)       # sessions outside the group covering each cell
    for c in cov:
        R0[c] += 1

    def gain(c, R, TP):
        # adding c: held-out points in cells only c covered stop overlapping; c's own points
        # overlap where some other outside session also covers them
        return TP[cov[c]][R[cov[c]] == 1].sum(), np.count_nonzero(R[pts[c]] > 1)

    ok = np.array([eligible is None or s in eligible for s in ids])
    found = {t: {} for t in targets}      # target -> {group: raster overlap}
    for start in np.flatnonzero(ok):
        R, TP = R0.copy(), np.zeros(len(allk), dtype=np.int32)
        group, ov, wt, cand = [], 0, 0.0, start
        pending = sorted(targets)
        while pending:
            freed, own = gain(cand, R, TP)
            R[cov[cand]] -= 1
            np.add.at(TP, pts[cand], 1)
            group.append(cand)
            ov, wt = ov - freed + own, wt + w[cand]
            while pending and wt / total >= pending[0]:
                t = pending.pop(0)
                found[t][tuple(sorted(ids[g] for g in group))] = ov / wt
            if not pending:
                break
            scores = np.full(len(ids), np.inf)
            for c in np.flatnonzero(ok):
                if c not in group:
                    fr, on = gain(c, R, TP)
                    scores[c] = (ov - fr + on) / (wt + w[c])
            if not np.isfinite(scores).any():
                break
            cand = int(np.argmin(scores))
    # every distinct group, lowest raster overlap first: the first is the split, and "the next
    # start" (if the first fails a pre-registered check) means the next one in this list
    return {t: sorted(((v, list(g)) for g, v in f.items()), key=lambda x: (x[0], x[1])) for t, f in found.items()}


def describe(group, P, M, V, sess):
    """Aggregates for a set of sessions: moving hours, days, Kish effective number of sessions,
    share of moving frames with |steer| >= 15 deg (from week1_checks sessions.csv) and of fast
    (> 25 m/s, roughly highway) moving points."""
    w = np.array([M[s].sum() for s in group], float)
    fast = sum((V[s][M[s]] > 25).sum() for s in group) / w.sum()
    out = {"sessions": len(group), "moving_hours": round(w.sum() / 3600, 2), "days": len({s[:8] for s in group}),
           "kish_sessions": round(w.sum() ** 2 / (w ** 2).sum(), 1), "fast_share": round(float(fast), 3)}
    if sess is not None:
        rows = [sess[s] for s in group if s in sess]
        n = sum(r[0] for r in rows)
        out["curve15_share"] = round(sum(r[0] * r[1] for r in rows) / n, 3) if n else None
    return out


def route_clusters(ids, P, M, r, cell, thr):
    """Group sessions that drive the same roads: an edge when either session has more than `thr`
    of its usable points within `r` of the other, then connected components. The held-out group in
    design A is chosen for mutual overlap, so its sessions are not independent samples; this says
    how many independent route groups it actually contains."""
    k = int(np.ceil(r / cell))
    off = np.array([(dx, dy) for dx in range(-k, k + 1) for dy in range(-k, k + 1)], dtype=np.int64)
    key = lambda c: (c[:, 0] + (1 << 24)) * (1 << 25) + (c[:, 1] + (1 << 24))
    cover, pts = {}, {}
    for i, s in enumerate(ids):
        c = np.floor(P[s] / cell).astype(np.int64)
        for cl in np.unique(key((np.unique(c, axis=0)[:, None, :] + off[None]).reshape(-1, 2))):
            cover.setdefault(cl, set()).add(i)
        pts[i] = key(c[M[s]])
    n = len(ids)
    share = np.zeros((n, n))
    for i in range(n):
        if not len(pts[i]):
            continue
        counts = np.zeros(n)
        for cl in pts[i]:
            for j in cover.get(cl, ()):
                counts[j] += 1
        share[i] = counts / len(pts[i])
    adj = (np.maximum(share, share.T) > thr) & ~np.eye(n, dtype=bool)
    seen, comps = set(), []
    for i in range(n):
        if i in seen:
            continue
        stack, comp = [i], []
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            comp.append(v)
            stack += list(np.flatnonzero(adj[v]))
        comps.append(sorted(comp))
    return comps, share


def trn_overlap(P, M, split, r):
    tr = [s for s in split["train_session_set"] if s in P]
    te = [s for s in split["test_session_set"] if s in P]
    tree = cKDTree(np.vstack([P[s] for s in tr]))
    per, hit, n = [], 0, 0
    for s in te:
        d, _ = tree.query(P[s][M[s]], distance_upper_bound=r)
        h = np.isfinite(d)
        per.append(h.mean())
        hit, n = hit + h.sum(), n + h.size
    per = np.array(per)
    return {"train_sessions": len(tr), "test_sessions": len(te), "per_session_mean": float(per.mean()),
            "pooled": float(hit / n), "test_sessions_over_half": int((per > 0.5).sum())}


def flat(ids, P, M):
    allp = np.vstack([P[s] for s in ids])
    mv = np.concatenate([M[s] for s in ids])
    owner = np.concatenate([np.full(len(P[s]), i) for i, s in enumerate(ids)])
    pos = np.concatenate([np.arange(len(P[s])) for s in ids])        # index within its session
    return allp, mv, owner, pos


def apply_rules(allp, mv, pos, lab, radius, windows):
    """The rules every split design gets: val points within `radius` of train are dropped, test
    points within `radius` of train or val are dropped, and a moving target point is kept only if
    it and the W points before it (same session) share its split and none of them is dropped."""
    tr, va = cKDTree(allp[lab == 0]), cKDTree(allp[lab == 1])
    drop = np.zeros(len(allp), bool)
    iv = np.flatnonzero(lab == 1)
    drop[iv[np.isfinite(tr.query(allp[iv], distance_upper_bound=radius)[0])]] = True
    it = np.flatnonzero(lab == 2)
    near = (np.isfinite(tr.query(allp[it], distance_upper_bound=radius)[0]) |
            np.isfinite(va.query(allp[it], distance_upper_bound=radius)[0]))
    drop[it[near]] = True
    out = {}
    for W in windows:
        keep = mv & ~drop
        for k in range(1, W + 1):
            prev = np.roll(np.arange(len(allp)), k)
            keep &= (pos >= k) & (lab[prev] == lab) & ~drop[prev]
        out[W] = keep
    return out


def kish(n):
    n = n[n > 0]
    return float(n.sum() ** 2 / (n ** 2).sum()) if n.size else 0.0


COLS = ["train", "val", "test", "test_sessions", "test_units", "kish_units", "kish_sessions", "sessions_train_and_test"]


def summarize(keep, lab, mv, owner, unit):
    kt = keep & (lab == 2)
    both = np.intersect1d(np.unique(owner[keep & (lab == 0)]), np.unique(owner[kt]))
    return [keep[lab == 0].sum() / mv.sum(), keep[lab == 1].sum() / mv.sum(), kt.sum() / mv.sum(),
            len(np.unique(owner[kt])), int((np.bincount(unit[kt]) > 0).sum()),
            kish(np.bincount(unit[kt])), kish(np.bincount(owner[kt])), len(both)]


def block_draws(ids, P, M, B, radius, windows, draws):
    allp, mv, owner, pos = flat(ids, P, M)
    _, blk = np.unique(np.floor(allp / B).astype(np.int64), axis=0, return_inverse=True)
    blk = blk.ravel()
    nb = blk.max() + 1
    rows = {W: [] for W in windows}
    for d in range(draws):
        u = np.random.default_rng(1000 + d).random(nb)
        lab = np.where(u < 0.63, 0, np.where(u < 0.75, 1, 2))[blk]
        for W, keep in apply_rules(allp, mv, pos, lab, radius, windows).items():
            rows[W].append(summarize(keep, lab, mv, owner, blk))
    return nb, {W: np.array(v) for W, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--split", help="TRN.pytorch data/data_info.json")
    ap.add_argument("--exclude", nargs="*", default=[], help="sessions with unreliable GPS")
    ap.add_argument("--radius", type=float, nargs="+", default=[50, 200])
    ap.add_argument("--blocks", type=float, nargs="+", default=[1000, 2000, 4000])
    ap.add_argument("--windows", type=int, nargs="+", default=[0, 2, 5], help="history, seconds")
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--cluster-threshold", type=float, default=0.5,
                    help="two sessions share a route group when either has this share of its points near the other")
    ap.add_argument("--json", help="also write the aggregates here")
    ap.add_argument("--sessions-csv", help="week1_checks sessions.csv, for the curve share")
    a = ap.parse_args()
    if a.json:
        refuse_inside_git(a.json)
    ids, P, M, V = load(a.tracks, set(a.exclude))
    nmov = int(sum(M[s].sum() for s in ids))
    out = {"sessions": len(ids), "moving_points": nmov,
           "points": int(sum(len(P[s]) for s in ids))}
    print(f"sessions with usable GPS: {len(ids)}; 1 Hz points: {out['points']}, moving: {nmov}")

    r0 = a.radius[0]
    a_radius, a_windows = a.radius, a.windows
    own = session_overlap(ids, P, M, r0)
    out["A_session_overlap"] = {"p10_p50_p90": np.nanpercentile(own, [10, 50, 90]).round(3).tolist(),
                                "no_moving_points": int(np.isnan(own).sum()),
                                "under_10pct": int((own < 0.1).sum()), "under_25pct": int((own < 0.25).sum())}
    print(f"A. share of a session's moving points within {r0:.0f} m of another session: "
          f"p10/p50/p90 {out['A_session_overlap']['p10_p50_p90']}, <10%: {out['A_session_overlap']['under_10pct']}, "
          f"<25%: {out['A_session_overlap']['under_25pct']}")

    ranked = greedy_holdout(ids, P, M, r0, cell=25.0, targets=[0.12, 0.25])
    best = {t: v[0] for t, v in ranked.items()}
    out["B_group_holdout"] = {}
    for t, (approx, group) in best.items():
        ex = exact_group_overlap(ids, P, M, group, r0)
        out["B_group_holdout"][str(t)] = {"sessions": len(group), "raster_overlap": round(float(approx), 3),
                                          "exact_overlap": round(ex, 3), "distinct_groups": len(ranked[t]),
                                          "next_raster_overlaps": [round(float(v), 3) for v, _ in ranked[t][1:4]]}
        print(f"B. best whole-session group holding out >= {t:.0%} of moving points: {len(group)} sessions, "
              f"overlap with the rest within {r0:.0f} m: {ex:.3f} (raster bound {approx:.3f}); "
              f"{len(ranked[t])} distinct groups, next raster overlaps {out['B_group_holdout'][str(t)]['next_raster_overlaps']}")

    sess = None
    if a.sessions_csv:
        import ast
        import pandas as pd
        df = pd.read_csv(a.sessions_csv, dtype={"session": str})
        df = df[df["steer_frames.bins_moving"].notna()]
        sess = {r.session: (r["steer_frames.n_moving"], ast.literal_eval(r["steer_frames.bins_moving"])[2])
                for _, r in df.iterrows()}

    # a whole-session split: test group first, then a val group chosen from the remaining sessions,
    # scored against everything that is not val (train and test)
    test = best[0.25][1]
    val = greedy_holdout(ids, P, M, r0, cell=25.0, targets=[0.12], eligible=set(ids) - set(test))[0.12][0][1]
    train = [s for s in ids if s not in test and s not in val]
    ws = {"train": describe(train, P, M, V, sess), "val": describe(val, P, M, V, sess),
          "test": describe(test, P, M, V, sess)}
    for r in a.radius:
        rest_t = cKDTree(np.vstack([P[s] for s in train + val]))
        rest_v = cKDTree(np.vstack([P[s] for s in train + test]))
        qt = np.vstack([P[s][M[s]] for s in test])
        qv = np.vstack([P[s][M[s]] for s in val])
        ws["test"][f"within_{r:.0f}m_of_train_or_val"] = round(float(np.isfinite(rest_t.query(qt, distance_upper_bound=r)[0]).mean()), 3)
        ws["val"][f"within_{r:.0f}m_of_train_or_test"] = round(float(np.isfinite(rest_v.query(qv, distance_upper_bound=r)[0]).mean()), 3)
    ws["all"] = describe(ids, P, M, V, sess)
    ws["days_shared_train_test"] = len({s[:8] for s in train} & {s[:8] for s in test})
    out["B_whole_session_split"] = ws
    print("B2. whole-session split (greedy test group, then val group):")
    for k in ("train", "val", "test", "all"):
        print(f"    {k:5s} {ws[k]}")
    print(f"    days with sessions in both train and test: {ws['days_shared_train_test']}")
    allp, mv, owner, pos = flat(ids, P, M)
    code = {s: 2 for s in test} | {s: 1 for s in val}
    labA = np.array([code.get(s, 0) for s in ids])[owner]
    ws["same_rules"] = []
    print("    same rules as the block splits (shares of all moving points; units = sessions):")
    print("      R m  W s | train  val  test | test sessions  Kish sessions  sessions in train & test")
    for r in a_radius:
        for W, keep in apply_rules(allp, mv, pos, labA, r, a_windows).items():
            v = summarize(keep, labA, mv, owner, owner)
            ws["same_rules"].append({"radius_m": r, "window_s": W, **dict(zip(COLS, np.round(v, 4).tolist()))})
            print(f"     {r:4.0f} {W:4d} | {v[0]:.3f}  {v[1]:.3f}  {v[2]:.3f} | {v[3]:13.0f}  {v[6]:13.1f}  {v[7]:10.0f}")

    # for comparison: whole sessions assigned at random (new drives, but mostly familiar roads)
    rnd = []
    for d in range(a.draws):
        u = np.random.default_rng(2000 + d).random(len(ids))
        labR = np.where(u < 0.63, 0, np.where(u < 0.75, 1, 2))
        te = [s for s, l in zip(ids, labR) if l == 2]
        rest = cKDTree(np.vstack([P[s] for s, l in zip(ids, labR) if l != 2]))
        q = np.vstack([P[s][M[s]] for s in te])
        rnd.append([len(q) / nmov, float(np.isfinite(rest.query(q, distance_upper_bound=r0)[0]).mean())])
    rnd = np.array(rnd)
    out["E_random_sessions"] = {"test_share_mean": float(rnd[:, 0].mean()),
                                "test_within_r0_of_rest_mean": float(rnd[:, 1].mean()),
                                "test_within_r0_of_rest_min": float(rnd[:, 1].min())}
    print(f"E. random whole-session splits: test share {rnd[:, 0].mean():.3f}, test points within {r0:.0f} m of "
          f"train or val {rnd[:, 1].mean():.3f} (min {rnd[:, 1].min():.3f}) over {a.draws} draws")

    # how many independent route groups the held-out sessions really are
    comps, share = route_clusters(ids, P, M, r0, 25.0, a.cluster_threshold)
    w = np.array([M[s].sum() for s in ids], float)
    for name, group in (("test", test), ("val", val), ("all", ids)):
        idx = [ids.index(s) for s in group]
        sub = [[i for i in c if i in idx] for c in comps]
        sub = [c for c in sub if c]
        sizes = sorted((len(c) for c in sub), reverse=True)
        cw = np.array([w[c].sum() for c in sub])
        out.setdefault("B_route_clusters", {})[name] = {
            "sessions": len(idx), "clusters": len(sub), "largest_clusters": sizes[:5],
            "kish_clusters": round(kish(cw), 1),
            "largest_cluster_share": round(float(cw.max() / cw.sum()), 3)}
        print(f"    route clusters in {name}: {len(sub)} clusters over {len(idx)} sessions, "
              f"largest {sizes[:5]}, Kish {kish(cw):.1f}, largest holds {cw.max() / cw.sum():.0%} of its data")

    if a.split:
        split = json.load(open(a.split))["HDD"]
        out["C_trn"] = {str(r): trn_overlap(P, M, split, r) for r in a.radius}
        for r, v in out["C_trn"].items():
            print(f"C. TRN split, test moving points within {float(r):.0f} m of train: per-session mean "
                  f"{v['per_session_mean']:.3f}, pooled {v['pooled']:.3f}, sessions over half "
                  f"{v['test_sessions_over_half']}/{v['test_sessions']} (train sessions with GPS: {v['train_sessions']})")

    out["D_blocks"] = []
    print("D. block splits; shares are of all moving points; mean (min) over draws")
    print("   B km  R m  W s | train  val (min)  test (min) | test sessions  test blocks  Kish blocks  Kish sessions  sessions in train & test")
    for B in a.blocks:
        for r in a.radius:
            nb, rows = block_draws(ids, P, M, B, r, a.windows, a.draws)
            for W, v in rows.items():
                m, lo = v.mean(0), v.min(0)
                out["D_blocks"].append({"block_m": B, "radius_m": r, "window_s": W, "occupied_blocks": int(nb),
                                        "cols": COLS, "mean": m.round(4).tolist(), "min": lo.round(4).tolist()})
                print(f"   {B / 1000:4.0f} {r:4.0f} {W:4d} | {m[0]:.3f}  {m[1]:.3f} ({lo[1]:.3f})  {m[2]:.3f} ({lo[2]:.3f}) | "
                      f"{m[3]:13.0f}  {m[4]:11.0f}  {m[5]:11.1f}  {m[6]:13.1f}  {m[7]:10.0f}   [{nb} occupied blocks]")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
