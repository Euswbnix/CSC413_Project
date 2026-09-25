"""Frozen visual features for HDD, one file per session. Statistics only leave the server.

Per session this decodes the front-camera video once and writes, to <out>/<session>.npz:

  feats      float16 (n_kept, D)  encoder features for the kept frames (--stride)
  kept       int32   (n_kept,)    index of each kept frame within the session
  t_ros      float64 (n_frames,)  receive timestamp of every decoded frame
  t_cam      float64 (n_frames,)  camera clock, unwrapped (128 s wrap), for real dt
  steer      float32 (n_frames,)  steering angle at that frame, NaN across CAN gaps > 50 ms
  speed_mps  float32 (n_frames,)  CAN speed converted to m/s, NaN likewise
  shift_px   float32 (n_frames,)  horizontal image motion against the previous frame

`shift_px` is a by-product of decoding: it is the lag that best aligns the column-intensity
profile of consecutive frames. Cross-correlating it with the CAN yaw rate is how we check that
frames and steering really are aligned, without anyone looking at a frame.

    python hdd/extract_features.py --videos ~/workspace/hdd/video --raw ~/workspace/hdd/raw \
        --out ~/workspace/hdd/features/dinov2_s10 --stride 10 --workers 4
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
WRAP = 128.0
MAX_CAN_GAP = 0.05
SKIP_START = 6
PROF_STEP = 8          # subsample step for the cheap motion profile
MAX_SHIFT = 40         # search range in profile columns


def refuse_inside_git(path):
    d = os.path.abspath(path if os.path.isdir(path) else os.path.dirname(path) or ".")
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            sys.exit(f"refusing to write HDD-derived output inside a git working tree ({d})")
        parent = os.path.dirname(d)
        if parent == d:
            return
        d = parent


def read_csv(path, cols):
    import pandas as pd
    keep = [i for i, c in enumerate(cols) if c not in ("iso", "path")]
    df = pd.read_csv(path, skiprows=1, header=None, usecols=keep, engine="c")
    df.columns = [cols[i] for i in keep]
    return df


def resample(t_src, v_src, t_dst, max_gap):
    ok = np.isfinite(v_src)
    ts, vs = t_src[ok], v_src[ok]
    out = np.interp(t_dst, ts, vs, left=np.nan, right=np.nan)
    i = np.clip(np.searchsorted(ts, t_dst), 1, ts.size - 1)
    out[(ts[i] - ts[i - 1]) > max_gap] = np.nan
    return out


def session_labels(raw, session, t_ros, speed_ratio):
    """Steering and speed at the frame times, never interpolated across a CAN gap."""
    import glob
    csv = glob.glob(os.path.join(raw, "release_2019_07_08", "*", session, "general", "csv"))[0]
    st = read_csv(os.path.join(csv, "steer.csv"), ["t", "iso", "angle", "speed"])
    ve = read_csv(os.path.join(csv, "vel.csv"), ["t", "iso", "v"])
    steer = resample(st["t"].to_numpy(float), st["angle"].to_numpy(float), t_ros, MAX_CAN_GAP)
    speed = resample(ve["t"].to_numpy(float), ve["v"].to_numpy(float), t_ros, MAX_CAN_GAP) / speed_ratio
    return steer.astype(np.float32), speed.astype(np.float32)


def frame_times(raw, session):
    """ROS timestamps and the unwrapped camera clock for every recorded frame."""
    import glob
    p = glob.glob(os.path.join(raw, "release_2019_07_25", "*", session, "camera", "center",
                               "png_timestamp.csv"))[0]
    d = read_csv(p, ["t", "iso", "path", "clk", "cnt"])
    t, clk = d["t"].to_numpy(float), d["clk"].to_numpy(float)
    return t, clk + WRAP * np.concatenate([[0], np.cumsum(np.diff(clk) < -WRAP / 2)])


def best_shift(a, b):
    """Columns that best align profile b onto a, by cross-correlation, 0 if degenerate."""
    a = a - a.mean()
    b = b - b.mean()
    if not (a.any() and b.any()):
        return 0.0
    c = np.correlate(a, b, mode="full")
    lo = len(b) - 1 - MAX_SHIFT
    return float(np.argmax(c[lo:lo + 2 * MAX_SHIFT + 1]) - MAX_SHIFT)


def build_encoder(name, device):
    import torch
    if name.startswith("dinov2"):
        os.environ.setdefault("XFORMERS_DISABLED", "1")     # flash-attn fails on sm_120
        m = torch.hub.load("facebookresearch/dinov2", name, verbose=False)
        def fwd(x):
            o = m.forward_features(x)
            return torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)], -1)
    elif name == "resnet50":
        import torchvision
        m = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        m.fc = torch.nn.Identity()
        fwd = m
    else:
        raise SystemExit(f"unknown encoder {name}")
    return m.eval().to(device).half(), fwd


def run_session(args, session, video):
    import cv2
    import torch
    cv2.setNumThreads(args.cv_threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, fwd = build_encoder(args.encoder, dev)
    mean = torch.tensor(MEAN, device=dev).half().view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).half().view(1, 3, 1, 1)

    t_ros, t_cam = frame_times(args.raw, session)
    steer, speed = session_labels(args.raw, session, t_ros, args.speed_ratio)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    feats, kept, shifts, batch, idx = [], [], [], [], 0
    prev_prof = None
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        prof = frame[::PROF_STEP, ::PROF_STEP].mean(axis=(0, 2))       # column profile, cheap
        shifts.append(0.0 if prev_prof is None else best_shift(prev_prof, prof))
        prev_prof = prof
        if idx >= SKIP_START and (idx - SKIP_START) % args.stride == 0:
            small = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
            batch.append(small[:, :, ::-1])
            kept.append(idx)
            if len(batch) == args.batch:
                feats.append(encode(batch, fwd, dev, mean, std))
                batch = []
        idx += 1
    if batch:
        feats.append(encode(batch, fwd, dev, mean, std))
    cap.release()

    n = idx
    if n != len(t_ros):
        print(f"  {session}: WARNING {n} decoded frames but {len(t_ros)} timestamp rows", flush=True)
    m = min(n, len(t_ros))
    out = pathlib.Path(args.out) / f"{session}.npz"
    np.savez(out, feats=np.concatenate(feats) if feats else np.zeros((0, 0), np.float16),
             kept=np.array(kept, np.int32), t_ros=t_ros[:m], t_cam=t_cam[:m],
             steer=steer[:m], speed_mps=speed[:m], shift_px=np.array(shifts[:m], np.float32))
    os.chmod(out, 0o600)
    dt = time.perf_counter() - t0
    print(f"  {session}: {n} frames, kept {len(kept)}, {n / dt:.0f} frames/s", flush=True)
    return n, len(kept), dt


def encode(batch, fwd, dev, mean, std):
    import torch
    x = torch.from_numpy(np.ascontiguousarray(np.stack(batch))).to(dev)
    x = x.permute(0, 3, 1, 2).half().div_(255)
    with torch.no_grad():
        return fwd((x - mean) / std).float().cpu().numpy().astype(np.float16)


def worker(args, sessions):
    done = frames = 0
    for s in sessions:
        video = next(pathlib.Path(args.videos).glob(f"{s}*.mp4"), None)
        if video is None:
            print(f"  {s}: no video", flush=True)
            continue
        if (pathlib.Path(args.out) / f"{s}.npz").exists() and not args.overwrite:
            continue
        n, _, _ = run_session(args, s, video)
        done, frames = done + 1, frames + n
    return done, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="directory of <session>*.mp4")
    ap.add_argument("--raw", required=True, help="extracted CAN/timestamp CSVs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="dinov2_vitb14")
    ap.add_argument("--stride", type=int, default=10, help="keep every k-th frame")
    ap.add_argument("--width", type=int, default=392)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cv-threads", type=int, default=2)
    ap.add_argument("--speed-ratio", type=float, default=3.6, help="CAN speed units per m/s")
    ap.add_argument("--sessions", nargs="*", help="default: every video found")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    refuse_inside_git(a.out)
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    os.chmod(a.out, 0o700)

    videos = sorted(pathlib.Path(a.videos).glob("*.mp4"))
    sessions = a.sessions or sorted({p.stem.split("_")[0] for p in videos})
    print(f"{len(sessions)} sessions, encoder {a.encoder}, stride {a.stride}, {a.workers} workers")
    (pathlib.Path(a.out) / "config.json").write_text(json.dumps(
        {k: v for k, v in vars(a).items() if k != "sessions"}, indent=1) + "\n")

    t0 = time.perf_counter()
    if a.workers == 1:
        worker(a, sessions)
    else:
        import multiprocessing as mp
        chunks = [sessions[i::a.workers] for i in range(a.workers)]
        with mp.get_context("spawn").Pool(a.workers) as pool:
            pool.starmap(worker, [(a, c) for c in chunks])
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
