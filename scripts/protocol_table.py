#!/usr/bin/env python3
"""Decompose the val->test drop into a PROTOCOL effect and a SPLIT effect.

`train.py` selects the checkpoint on WINDOWED val macro skill (state reset every T frames);
`evaluate.py` reports ROLLOUT test (state carried across the split). Those are two changes
at once, so the reported drop has never been attributable to either. Adding rollout-val
closes the 2x2:

                   windowed              rollout
    val         selection metric      protocol-matched   <- new
    test               --             REPORTED NUMBER

Protocol effect = rollout-val  - windowed-val   (same split, protocol changes)
Split effect    = rollout-test - rollout-val    (same protocol, split changes)

The three numbers live in three files, because they were produced by two programs:
  windowed-val  : final_metrics.json      ["val"]      written by train.py
  rollout-val   : final_metrics_val.json  ["rollout"]  written by evaluate.py --split val
  rollout-test  : final_metrics_test.json ["rollout"]  written by evaluate.py --split test
"""
import json, pathlib, sys
import numpy as np

SNAP = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "lab_snapshot")
VAL  = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "runs_val")
ARMS = ("cnn_linear", "cnn_2frame", "cnn_mlp", "cnn_avg", "lstm_dt", "lstm", "cfc", "gru")


def arm_of(n):
    return next((a for a in ARMS if n.startswith(a)), n.split("_")[0])


def jload(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def collect():
    out = {}
    for p in SNAP.rglob("final_metrics.json"):
        d = jload(p)
        if d and isinstance(d.get("val"), dict):
            out.setdefault(p.parent.name, {})["win_val"] = d["val"]
    for p in SNAP.rglob("final_metrics_test.json"):
        d = jload(p)
        if d and isinstance(d.get("rollout"), dict):
            out.setdefault(p.parent.name, {})["roll_test"] = d["rollout"]
    for p in VAL.rglob("final_metrics_val.json"):
        d = jload(p)
        if d and isinstance(d.get("rollout"), dict):
            out.setdefault(p.parent.name, {})["roll_val"] = d["rollout"]
            out[p.parent.name]["pos_delta"] = d.get("position_delta_deg")
    return out


def main():
    runs = collect()
    have = {k: v for k, v in runs.items() if {"win_val", "roll_val"} <= set(v)}
    by = {}
    for name, v in have.items():
        by.setdefault(arm_of(name), []).append(v)

    print(f"{len(runs)} runs seen; {len(have)} have both windowed-val and rollout-val\n")
    hdr = (f"{'arm':<11}{'n':>4}{'winVal':>9}{'rollVal':>9}{'rollTest':>10}"
           f"{'protocol':>10}{'split':>8}{'r winVal':>10}{'r rollVal':>11}{'posDelta':>10}")
    print(hdr); print("-" * len(hdr))

    def med(vs, blk, key):
        xs = [v[blk][key] for v in vs if blk in v and v[blk].get(key) is not None
              and not (isinstance(v[blk][key], float) and np.isnan(v[blk][key]))]
        return float(np.median(xs)) if xs else float("nan")

    for arm in sorted(by):
        vs = by[arm]
        wv = med(vs, "win_val", "macro_skill")
        rv = med(vs, "roll_val", "macro_skill")
        rt = med(vs, "roll_test", "macro_skill")
        rw = med(vs, "win_val", "pearson_r")
        rr = med(vs, "roll_val", "pearson_r")
        pd = [v["pos_delta"] for v in vs if v.get("pos_delta") is not None]
        pdm = float(np.median(pd)) if pd else float("nan")
        print(f"{arm:<11}{len(vs):>4}{wv:>+9.4f}{rv:>+9.4f}{rt:>+10.4f}"
              f"{rv-wv:>+10.4f}{rt-rv:>+8.4f}{rw:>+10.3f}{rr:>+11.3f}{pdm:>+10.2f}")

    print("\nprotocol = rollout-val − windowed-val   (同一 split，只换评估协议)")
    print("split    = rollout-test − rollout-val   (同一协议，只换 split)")
    print("posDelta = windowed 下 t=1 到 t=T 的 MAE 改善，度（正=更多上下文更准）")


if __name__ == "__main__":
    main()
