"""How fast can we turn HDD video into frozen features? Timings only, no frames are shown.

Runs the real stages on one video: decode with OpenCV, resize to the encoder's input, run a
frozen encoder in fp16 on the GPU, write float16 features. Reports frames per second per stage
and end to end, for keeping every frame and for keeping every k-th frame (decoding is the same
either way, which is the point). Extrapolates to a given total frame count.

    python hdd/bench_extract.py --video /path/to/one.mp4 --frames 2000 --encoder dinov2_vitb14
"""
import argparse
import pathlib
import time

import cv2
import numpy as np
import torch

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_encoder(name, device):
    if name.startswith("dinov2"):
        m = torch.hub.load("facebookresearch/dinov2", name, verbose=False)
        dim = m.embed_dim * 2                       # CLS + mean patch, concatenated
        def fwd(x):
            o = m.forward_features(x)
            return torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)], -1)
    elif name == "resnet50":
        import torchvision
        m = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        m.fc = torch.nn.Identity()
        dim, fwd = 2048, lambda x: m(x)
    else:
        raise SystemExit(f"unknown encoder {name}")
    return m.eval().to(device).half(), fwd, dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--encoder", default="dinov2_vitb14")
    ap.add_argument("--width", type=int, default=392)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--keep-every", type=int, default=10, help="with --subset, encode every k-th frame")
    ap.add_argument("--subset", action="store_true", help="decode everything, encode every k-th frame (the D2 run)")
    ap.add_argument("--gpu-resize", action="store_true", help="upload full frames and resize on the GPU")
    ap.add_argument("--total-frames", type=float, default=11_233_119, help="for the extrapolation")
    ap.add_argument("--out", default="/dev/shm/bench_feats.f16")
    ap.add_argument("--cv-threads", type=int, default=4, help="OpenCV threads; the server is shared")
    a = ap.parse_args()

    cv2.setNumThreads(a.cv_threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, fwd, dim = build_encoder(a.encoder, dev)
    mean, std = MEAN.to(dev).half(), STD.to(dev).half()
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")

    t = {k: 0.0 for k in ("decode", "resize", "infer", "write")}
    out = open(a.out, "wb")
    batch, n_dec, n_enc = [], 0, 0
    wall = time.perf_counter()
    while n_dec < a.frames:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t["decode"] += time.perf_counter() - t0
        if not ok:
            break
        n_dec += 1
        if a.subset and (n_dec - 1) % a.keep_every:
            continue                      # decoding still happened: that is what this measures
        t0 = time.perf_counter()
        if a.gpu_resize:
            batch.append(frame[:, :, ::-1])                    # BGR -> RGB, resize happens below
        else:
            small = cv2.resize(frame, (a.width, a.height), interpolation=cv2.INTER_AREA)
            batch.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        t["resize"] += time.perf_counter() - t0
        if len(batch) == a.batch:
            t0 = time.perf_counter()
            x = torch.from_numpy(np.ascontiguousarray(np.stack(batch))).to(dev, non_blocking=True)
            x = x.permute(0, 3, 1, 2).half().div_(255)
            if a.gpu_resize:
                x = torch.nn.functional.interpolate(x, (a.height, a.width), mode="bilinear",
                                                    align_corners=False, antialias=True)
            with torch.no_grad():
                f = fwd((x - mean) / std)
            torch.cuda.synchronize() if dev == "cuda" else None
            t["infer"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            out.write(f.cpu().numpy().astype(np.float16).tobytes())
            t["write"] += time.perf_counter() - t0
            n_enc += len(batch)
            batch = []
    total = time.perf_counter() - wall
    out.close()
    pathlib.Path(a.out).unlink(missing_ok=True)
    cap.release()

    print(f"encoder {a.encoder} ({dim}-d), input {a.width}x{a.height}, batch {a.batch}, device {dev}")
    print(f"decoded {n_dec} frames, encoded {n_enc}, wall {total:.1f}s -> {n_dec / total:.0f} frames/s end to end")
    for k, v in t.items():
        print(f"  {k:>7}: {v:6.1f}s ({v / total:4.0%})" + (f"  {n_enc / v:7.0f} enc frames/s" if k in ("infer", "write") and v else
                                                           f"  {n_dec / v:7.0f} frames/s" if v else ""))
    hours = a.total_frames / (n_dec / total) / 3600
    print(f"extrapolated to {a.total_frames:,.0f} frames: {hours:.1f} h at this rate (one pass, one process)")


if __name__ == "__main__":
    main()
