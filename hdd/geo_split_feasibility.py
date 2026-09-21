"""How much road overlap is unavoidable in an HDD split, at session level and at block level.

Reads checks/tracks_1hz.npz from week1_checks.py. Prints aggregates only.
  * session level: for each session, the share of its points within 50 m of any *other* session
  * block level: assign square blocks of side B at random to train/val/test (seeded), then report
    the test share, and how much test survives after dropping points within 50 m of train
"""
import argparse

import numpy as np
from scipy.spatial import cKDTree

EARTH_R = 6_371_000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="sessions with unreliable GPS")
    ap.add_argument("--blocks", type=float, nargs="+", default=[1000, 2000, 4000])
    ap.add_argument("--draws", type=int, default=20)
    a = ap.parse_args()

    z = np.load(a.tracks)
    ids = [s for s in z.files if s not in a.exclude]
    lat0 = np.median(np.concatenate([z[s][:, 0] for s in ids]))
    lon0 = np.median(np.concatenate([z[s][:, 1] for s in ids]))

    def xy(ll):
        return np.column_stack([np.radians(ll[:, 1] - lon0) * EARTH_R * np.cos(np.radians(lat0)),
                                np.radians(ll[:, 0] - lat0) * EARTH_R])

    P = {s: xy(z[s]) for s in ids}
    allp = np.vstack([P[s] for s in ids])
    owner = np.concatenate([np.full(len(P[s]), i) for i, s in enumerate(ids)])
    print(f"sessions with usable GPS: {len(ids)}; points (1 Hz): {len(allp)}; "
          f"extent {np.ptp(allp[:, 0]) / 1000:.0f} x {np.ptp(allp[:, 1]) / 1000:.0f} km")

    tree = cKDTree(allp)
    own = []
    for i, s in enumerate(ids):
        hits = tree.query_ball_point(P[s], r=50)
        own.append(np.mean([np.any(owner[h] != i) for h in hits]))
    own = np.array(own)
    print("session level: share of a session's points within 50 m of another session's track")
    print("  p10/p50/p90:", np.percentile(own, [10, 50, 90]).round(3),
          " sessions with <10% shared:", int((own < 0.10).sum()), " <25%:", int((own < 0.25).sum()))

    for B in a.blocks:
        cell = np.floor(allp / B).astype(np.int64)
        keys, inv = np.unique(cell, axis=0, return_inverse=True)
        inv = inv.ravel()
        res = []
        for d in range(a.draws):
            rng = np.random.default_rng(1000 + d)
            u = rng.random(len(keys))
            lab = np.where(u < 0.63, 0, np.where(u < 0.75, 1, 2))[inv]     # 0 train, 1 val, 2 test
            tr = cKDTree(allp[lab == 0])
            te = allp[lab == 2]
            dist, _ = tr.query(te, distance_upper_bound=50)
            kept = te[np.isinf(dist)]
            res.append((np.mean(lab == 2), len(kept) / len(allp),
                        len(np.unique(owner[lab == 2][np.isinf(dist)]))))
        r = np.array(res)
        print(f"blocks {B / 1000:.0f} km ({len(keys)} occupied): test share {r[:, 0].mean():.3f}, "
              f"test kept after 50 m buffer {r[:, 1].mean():.3f} of all points "
              f"(min {r[:, 1].min():.3f}), sessions contributing to test {r[:, 2].mean():.0f}")


if __name__ == "__main__":
    main()
