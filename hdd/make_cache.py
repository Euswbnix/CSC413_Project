"""Write the base-rate features once as plain .npy so parallel runs share one page cache.

The per-session npz files are compressed, so every process that opens them decompresses into its
own memory: six seeds in parallel would need six copies of the same 7 GB. This writes the
base-rate slice as uncompressed arrays plus a small index, which np.load(mmap_mode="r") maps
straight from the page cache.

    python hdd/make_cache.py --features ~/workspace/hdd/features/dinov2_s1 --split ~/workspace/hdd/split_a \
        --out ~/workspace/hdd/cache/dinov2_10hz --base-hz 10

--add appends sessions of the given --splits to an existing cache without rewriting (or even
reopening) the ones already there, e.g. the test split once, for the single test evaluation:

    python hdd/make_cache.py ... --splits test --add
"""
import argparse
import json
import os
import sys

import numpy as np


def refuse_inside_git(path):
    d = os.path.abspath(path if os.path.isdir(path) else os.path.dirname(path) or ".")
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            sys.exit(f"refusing to write HDD-derived output inside a git working tree ({d})")
        parent = os.path.dirname(d)
        if parent == d:
            return
        d = parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-hz", type=float, default=10.0)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--add", action="store_true", help="append to an existing cache; never touch its sessions")
    a = ap.parse_args()
    refuse_inside_git(a.out)
    os.makedirs(a.out, exist_ok=True)
    os.chmod(a.out, 0o700)
    info = json.load(open(os.path.join(a.split, "split.json")))["sessions"]
    stride = max(1, int(round(30.0 / a.base_hz)))
    index = {}
    if a.add:
        old = json.load(open(os.path.join(a.out, "index.json")))
        if (old["base_hz"], old["stride"], old["features"]) != (a.base_hz, stride, os.path.basename(a.features)):
            sys.exit(f"--add: the cache was built with {old['base_hz']} Hz / stride {old['stride']} / "
                     f"{old['features']}; refusing to mix")
        for s, meta in old["sessions"].items():
            if (info[s]["split"], info[s]["cluster"]) != (meta["split"], meta["cluster"]):
                sys.exit(f"--add: {s} is {meta['split']}/{meta['cluster']} in the cache but "
                         f"{info[s]['split']}/{info[s]['cluster']} in split.json")
        index = dict(old["sessions"])
    for s, meta in sorted(info.items()):
        if meta["split"] not in a.splits or s in index:
            continue
        p = os.path.join(a.features, f"{s}.npz")
        if not os.path.exists(p):
            if a.add:                                   # an appended split must be complete
                sys.exit(f"--add: no features for {meta['split']} session {s}")
            continue
        z = np.load(p)
        kept = z["kept"]
        sel = np.arange(0, len(kept), stride)
        idx = kept[sel]
        np.save(os.path.join(a.out, f"{s}.feats.npy"), z["feats"][sel])
        np.save(os.path.join(a.out, f"{s}.meta.npy"),
                np.column_stack([z["t_cam"][idx], z["steer"][idx], z["speed_mps"][idx]]).astype(np.float64))
        index[s] = dict(split=meta["split"], cluster=meta["cluster"], n=int(len(sel)))
        os.chmod(os.path.join(a.out, f"{s}.feats.npy"), 0o600)
        os.chmod(os.path.join(a.out, f"{s}.meta.npy"), 0o600)
    tmp = os.path.join(a.out, "index.json.tmp")
    json.dump(dict(base_hz=a.base_hz, stride=stride, features=os.path.basename(a.features),
                   sessions=index), open(tmp, "w"), indent=1)
    os.chmod(tmp, 0o600)
    os.replace(tmp, os.path.join(a.out, "index.json"))      # a crash never leaves half an index
    gb = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out)) / 1e9
    from collections import Counter
    print(f"cached {len(index)} sessions at {a.base_hz:g} Hz ({dict(Counter(m['split'] for m in index.values()))}), "
          f"{gb:.1f} GB in {a.out}")


if __name__ == "__main__":
    main()
