"""Are the gaps in HDD's ROS frame timestamps dropped frames or receive-side delay? Aggregates only.

png_timestamp.csv has, per recorded frame, the ROS receive time and two camera-side columns:
a clock in seconds that wraps every 128 s, and a counter. For every session this script
  * unwraps the camera clock and fits ROS time = a + b * clock (b ~ 1 if the clock is real time),
  * compares each ROS gap (> 1.5 x median interval) with the camera-clock step over the same frames:
      clock step also long  -> a frame really was not recorded
      clock step normal     -> the frame arrived late (receive jitter), nothing is missing
  * counts clock gaps that the ROS timestamps do not show,
  * measures how far ROS time strays from the camera clock once a constant rate difference is
    fitted out, how long it strays by more than 0.2 s / 1 s, and whether it strays by a jump (a
    large step between consecutive frames) or gradually. Timestamps alone cannot say which clock
    is wrong: a receive backlog would misalign frames and CAN, a slewed host clock would not.
Usage: python hdd/timestamp_gaps.py --raw ~/data/hdd/raw --out ~/data/hdd/checks/timestamp_gaps.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

WRAP = 128.0
SKIP = 6            # start-up frames, as in week1_checks (at most 6)



def refuse_inside_git(path):
    """HDD-derived outputs must never land in a git working tree (licence §4.b; this repo is public)."""
    d = os.path.abspath(path if os.path.isdir(path) else os.path.dirname(path) or ".")
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            sys.exit(f"refusing to write HDD-derived output inside a git working tree ({d}); use ~/data/hdd/")
        parent = os.path.dirname(d)
        if parent == d:
            return
        d = parent

def session(path):
    d = pd.read_csv(path, skiprows=1, header=None, usecols=[0, 3, 4])
    t, clk, cnt = (d[c].to_numpy(float) for c in d.columns)
    t, clk, cnt = t[SKIP:], clk[SKIP:], cnt[SKIP:]
    step = np.diff(clk)
    wraps = step < -WRAP / 2
    clk_u = clk + WRAP * np.concatenate([[0], np.cumsum(wraps)])
    dt, dc = np.diff(t), np.diff(clk_u)
    b, a = np.polyfit(clk_u, t, 1)
    resid = t - (a + b * clk_u)            # ROS minus camera clock, after a constant-rate fit
    frame_w = np.diff(t, append=t[-1])     # seconds each frame stands for
    med_t, med_c = np.median(dt), np.median(dc)
    ros_gap, clk_gap = dt > 1.5 * med_t, dc > 1.5 * med_c
    return {
        "session": os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path)))),
        "frames": int(t.size), "counter_steps_not_1": int((np.diff(cnt) != 1).sum()),
        "clock_rate": float(b), "resid_ms_p50": float(np.percentile(np.abs(resid), 50) * 1e3),
        "resid_ms_p99": float(np.percentile(np.abs(resid), 99) * 1e3), "resid_ms_max": float(np.abs(resid).max() * 1e3),
        "clock_nonpos_steps": int((dc <= 0).sum()),
        "median_dt_ros": float(med_t), "median_dt_clock": float(med_c),
        "ros_gaps": int(ros_gap.sum()), "ros_gaps_also_clock_gap": int((ros_gap & clk_gap).sum()),
        "clock_gaps": int(clk_gap.sum()), "clock_gaps_not_in_ros": int((clk_gap & ~ros_gap).sum()),
        "missing_frames_clock": int(np.round(dc[clk_gap] / med_c).sum() - clk_gap.sum()),
        "missing_time_clock_s": float((dc[clk_gap] - med_c).sum()),
        "clock_gaps_over_1s": int((dc > 1.0).sum()),
        "span_s": float(t[-1] - t[0]),
        # separates a steady drift from a single jump: the largest change of the residual
        # between two consecutive frames
        "resid_max_step_s": float(np.abs(np.diff(resid)).max()),
        "time_resid_over_0p2s": float(frame_w[np.abs(resid) > 0.2].sum()),
        "time_resid_over_1s": float(frame_w[np.abs(resid) > 1.0].sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    refuse_inside_git(a.out)
    rows = [session(p) for p in sorted(glob.glob(os.path.join(a.raw, "release_2019_07_25", "*", "*", "camera", "center",
                                                             "png_timestamp.csv")))]
    df = pd.DataFrame(rows)
    tot = lambda c: int(df[c].sum())
    summary = {
        "sessions": len(df),
        "counter_steps_not_1": tot("counter_steps_not_1"),
        "clock_rate_p1_p50_p99": np.percentile(df.clock_rate, [1, 50, 99]).round(6).tolist(),
        "resid_ms_session_medians_p50": float(df.resid_ms_p50.median()),
        "resid_ms_p99_session_median": float(df.resid_ms_p99.median()),
        "resid_ms_max_max": float(df.resid_ms_max.max()),
        "clock_nonpos_steps": tot("clock_nonpos_steps"),
        "ros_gaps": tot("ros_gaps"), "ros_gaps_also_clock_gap": tot("ros_gaps_also_clock_gap"),
        "clock_gaps": tot("clock_gaps"), "clock_gaps_not_in_ros": tot("clock_gaps_not_in_ros"),
        "missing_frames_clock": tot("missing_frames_clock"),
        "missing_time_clock_h": float(df.missing_time_clock_s.sum() / 3600),
        "clock_gaps_over_1s": tot("clock_gaps_over_1s"),
        "sessions_with_clock_gap_over_1s": int((df.clock_gaps_over_1s > 0).sum()),
        "sessions_resid_over_0p2s": int((df.resid_ms_max > 200).sum()),
        "hours_of_those_sessions_0p2": float(df.loc[df.resid_ms_max > 200, "span_s"].sum() / 3600),
        "hours_with_resid_over_0p2s": float(df.time_resid_over_0p2s.sum() / 3600),
        "sessions_resid_over_1s": int((df.resid_ms_max > 1000).sum()),
        "hours_of_those_sessions_1": float(df.loc[df.resid_ms_max > 1000, "span_s"].sum() / 3600),
        "hours_with_resid_over_1s": float(df.time_resid_over_1s.sum() / 3600),
        "max_resid_step_in_drift_sessions_s": float(df.loc[df.resid_ms_max > 1000, "resid_max_step_s"].max()),
        "clock_gaps_not_in_ros_in_drift_sessions": int(df.loc[df.resid_ms_max > 1000, "clock_gaps_not_in_ros"].sum()),
        "worst_clock_rate_sessions": df.nsmallest(3, "clock_rate")[["session", "clock_rate"]].values.tolist()
                                     + df.nlargest(3, "clock_rate")[["session", "clock_rate"]].values.tolist(),
        "worst_resid_sessions": df.nlargest(5, "resid_ms_max")[["session", "resid_ms_max"]].values.tolist(),
    }
    json.dump({"summary": summary, "sessions": rows}, open(a.out, "w"), indent=1)
    # session IDs stay in the written file; stdout carries aggregates only
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("worst_")}, indent=1))


if __name__ == "__main__":
    main()
