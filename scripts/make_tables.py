#!/usr/bin/env python3
"""Emit the README's results tables from runs/, so they are generated output, not transcription.

    python scripts/make_tables.py --runs runs --split test > docs/results.md

One headline table. Rows are predict-0, persistence, a rule, then the models; columns are the
bins plus overall Pearson r, with each column header carrying that bin's test-frame count AND
its independent turn-event count -- 200 correlated frames from one corner are one sample, not
200, and a reader who cannot see that will over-read the table.

Each cell gives MAE in degrees and the same-bin ratio to predict-0, `3.9 (0.62x)`, so the
"must not lose to a naive baseline" question is answered without the reader dividing.
"""

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from metrics import BIN_NAMES


def load(runs, split):
    out = []
    for f in sorted(pathlib.Path(runs).glob(f"*/final_metrics_{split}.json")):
        out.append(json.loads(f.read_text()))
    return out


def group_by_arm(rows):
    g = {}
    for r in rows:
        g.setdefault(r["arm"], []).append(r)
    return g


def cell(mae, base):
    if mae != mae:
        return "--"
    return f"{mae:.2f} ({mae/base:.2f}x)" if base else f"{mae:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    rows = load(args.runs, args.split)
    if not rows:
        print(f"no `final_metrics_{args.split}.json` under {args.runs}/ -- run evaluate.py first",
              file=sys.stderr)
        raise SystemExit(1)

    cols = BIN_NAMES + ["(diagnostic) >=40"]
    ref = rows[0]["rollout"]["bins"]
    head = ["arm", "n seeds"] + [
        f"{c}<br><sub>n={ref[c]['n_frames']}, {ref[c]['n_events']} events</sub>" for c in cols
    ] + ["macro MAE", "macro skill", "Pearson r"]

    print(f"### Headline results, {args.split} split, stateful rollout\n")
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))

    b0 = rows[0]["rollout"]["bins"]
    print("| **predict-0** | -- | "
          + " | ".join(f"{b0[c]['mae_predict0']:.2f} (1.00x)" for c in cols)
          + f" | {rows[0]['rollout']['macro_mae_predict0']:.2f} | 0.000 | -- |")
    print("| **persistence**<sub>*</sub> | -- | "
          + " | ".join(cell(b0[c]["mae_persistence"], b0[c]["mae_predict0"]) for c in cols)
          + " | -- | -- | -- |")
    print("|" + " |" * (len(head) - 1) + " |")

    for arm, rs in sorted(group_by_arm(rows).items()):
        n = len(rs)
        med = lambda xs: statistics.median(xs) if xs else float("nan")
        cells = []
        for c in cols:
            vals = [r["rollout"]["bins"][c]["mae_model"] for r in rs]
            vals = [v for v in vals if v == v]
            cells.append(cell(med(vals), b0[c]["mae_predict0"]))
        mm = med([r["rollout"]["macro_mae"] for r in rs])
        ms = [r["rollout"]["macro_skill"] for r in rs]
        band = f" <sub>[{min(ms):+.3f}, {max(ms):+.3f}]</sub>" if n > 1 else ""
        print(f"| {arm} | {n} | " + " | ".join(cells)
              + f" | {mm:.2f} | {med(ms):+.3f}{band} | {med([r['rollout']['pearson_r'] for r in rs]):+.3f} |")

    print("\n<sub>*</sub> Persistence uses the previous TRUE angle. It is reported because it is"
          " the naive baseline a reader will think of, and it is not a solution to the posed"
          " task: the model receives images only and no past ground-truth angles, so"
          " persistence is unavailable the moment labels are absent -- which is always, at"
          " deployment.\n")
    print("Cells are MAE in degrees with the ratio to predict-0 in the same bin. Bins are"
          " assigned by GROUND TRUTH; metrics are per frame; invalid frames (a logging default"
          " of 0.0 in place of a real angle) are excluded. Bands over multiple seeds are"
          " min-max, not confidence intervals. The `>=40` column is a SUBSET of the curve bin,"
          " reported for diagnosis -- with 2 independent turn events in the test split it is"
          " not a measurement on its own.\n")

    print("### Per-position MAE (windowed, state reset at each window start)\n")
    print("| arm | t=1 | t=T | delta |")
    print("|---|---:|---:|---:|")
    for arm, rs in sorted(group_by_arm(rows).items()):
        c = rs[0]["mae_by_window_position"]
        if c and c[0] is not None:
            print(f"| {arm} | {c[0]:.2f} | {c[-1]:.2f} | {c[0]-c[-1]:+.2f} |")
    print("\nA non-recurrent arm must be FLAT here: that is the figure's own self-test.\n")

    print("### Runs\n\n| run | arm | seed | best epoch | params (recurrent) |")
    print("|---|---|---:|---:|---:|")
    for r in sorted(rows, key=lambda r: (r["arm"], r["seed"])):
        print(f"| `{r['run']}` | {r['arm']} | {r['seed']} | {r['best_epoch']} |"
              f" {r['params']['recurrent']:,} |")


if __name__ == "__main__":
    main()
