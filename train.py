#!/usr/bin/env python3
"""Train one arm. One run -> one directory under runs/, self-describing enough to reproduce.

    python train.py --arm cfc  --seed 0
    python train.py --arm lstm --seed 0
    python train.py --arm cfc  --seed 0 --fixed-dt        # question 2: does real dt help?
    python train.py --arm cfc  --seed 0 --shuffle-frames  # does the recurrent path carry load?

Three things are not options.

**The loss is masked.** 973 frames carry a label of exactly 0.0 recorded beside a large
value -- a logging default, not a centred wheel. The image is identical to its neighbours'
and the target is wrong, so the frame is unlearnable and MSE would punish the model for a
corruption it cannot see.

**Selection is on validation macro skill, never global MSE.** Global MSE picks the most
shrunken checkpoint of every run, which is exactly what per-bin metrics exist to prevent.
Both are logged every epoch so the counterfactual costs nothing.

**TF32 is off.** PyTorch's Blackwell defaults are asymmetric -- on for cuDNN convolutions,
off for cuBLAS matmuls -- so the encoder would run at a 10-bit mantissa while the CfC's
linear layers ran true fp32, an asymmetric precision difference inside the headline
comparison.
"""

import os

# MUST precede `import torch`: torch.use_deterministic_algorithms(True) makes every cuBLAS
# GEMM raise unless cuBLAS has been given a fixed workspace, and cuBLAS reads this variable
# once, when CUDA initialises. Setting it later has no effect. Leaving it to the caller's
# environment is what made determinism silently dependent on how the job was launched --
# the fleet shells happened to export it, so the failure only appeared on a host that did
# not. setdefault, so an explicitly chosen value still wins.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json
import pathlib
import random
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import metrics
from data.dataset import SteeringData, train_batches, window_batches
from models.interface import build_arm

FIXED = dict(optimizer="AdamW", weight_decay=1e-4, clip_grad_norm=1.0, batch_size=64,
             schedule="cosine", patience=10)


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def versions():
    import importlib.metadata as md
    out = {}
    for p in ("torch", "numpy", "ncps"):
        try:
            # importlib.metadata, NOT module.__version__: the ncps 1.0.1 wheel's __init__
            # declares "0.0.2", so every config.json would cite a version that does not exist.
            out[p] = md.version(p)
        except Exception:
            out[p] = None
    return out


def set_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def masked_mse(pred, y, valid):
    d = (pred.squeeze(-1) - y) ** 2 * valid
    return d.sum() / valid.sum().clamp(min=1)


def masked_bin_balanced_mse(pred, y, valid, mean, std):
    """Plain MSE, but with each PRIMARY bin contributing equally -- mirroring macro MAE.

    Measured on the train split: the straight bin is 42.6% of frames and carries 0.3% of the
    squared signal; the curve bin is 27.4% of frames and carries 96.7% (>=40 alone carries
    86.0%). macro MAE, which is the reported metric AND the selection metric, weights the
    three bins 1/3 each. So the objective and the yardstick have been measuring different
    quantities, and the mismatch lands precisely on the straight bin -- 55-61% of the
    evaluation frames, and the one bin where predict-0 is close to unbeatable.

    This changes the OBJECTIVE to match the pre-registered metric. It does not touch the
    metric. That direction matters: tuning the loss toward a metric fixed in advance is
    ordinary practice, whereas moving the bin edges to flatter the model would be reversing
    a pre-registered decision after seeing the results.

    `y` arrives standardised, so the edges are converted rather than the labels. Bins absent
    from a batch are dropped and the weights renormalised over those present, so the loss
    stays an average over bins rather than silently reweighting toward whatever is on hand.
    """
    d = (pred.squeeze(-1) - y) ** 2 * valid
    deg = y * std + mean
    ay = deg.abs()
    e0, e1 = metrics.BIN_EDGES[1], metrics.BIN_EDGES[2]   # 5.0, 15.0 -- [0] is 0.0
    terms = []
    for m in ((ay < e0), (ay >= e0) & (ay < e1), (ay >= e1)):
        m = (m & valid.bool()).to(d.dtype)
        n = m.sum()
        if n > 0:
            terms.append((d * m).sum() / n)
    if not terms:
        return d.sum() * 0.0
    return torch.stack(terms).mean()


def shuffle_within_windows(frames, y, valid, dt, gen):
    """Destroy frame ORDER inside each window, keeping the multiset of frames.

    The strongest single statement available about whether the recurrent pathway carries
    load: if a model trained on permuted windows lands inside the single-frame CNN's range,
    the recurrence was decorative. Eval-time shuffling is a weaker version of the same idea
    because the weights were still learned on ordered data.
    """
    B, T = y.shape
    perm = torch.argsort(torch.rand(B, T, device=frames.device, generator=gen), dim=1)
    g = perm[:, :, None, None, None].expand(-1, -1, *frames.shape[2:])
    return frames.gather(1, g), y.gather(1, perm), valid.gather(1, perm), dt.gather(1, perm)


@torch.no_grad()
def evaluate_windows(model, data, split, T, batch_size, fixed_dt):
    """Windowed validation: cheap, deterministic, and the same view the per-position figure
    uses. Final TEST numbers come from the stateful rollout in evaluate.py instead."""
    model.eval()
    P, Y, V = [], [], []
    for b in window_batches(data, split, T, batch_size):
        dt = torch.ones_like(b.dt) if fixed_dt else b.dt
        pred, _ = model(b.frames, dt=dt)
        P.append(pred.squeeze(-1).float().cpu())
        Y.append(b.y.float().cpu())
        V.append(b.valid.cpu())
    model.train()
    p = data.to_degrees(torch.cat(P)).numpy().ravel()
    y = data.to_degrees(torch.cat(Y)).numpy().ravel()
    v = torch.cat(V).numpy().ravel()
    mse_std = float(((torch.cat(P) - torch.cat(Y)) ** 2 * torch.cat(V)).sum()
                    / torch.cat(V).sum().clamp(min=1))
    return metrics.summary(p, y, v), mse_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=float, default=0.1, help="translation coefficient, deg/px")
    ap.add_argument("--aug", choices=("none", "basic", "full"), default="full",
                    help="none = no augmentation; basic = flip+brightness+shadow, k=0; "
                         "full = basic + translation with steering compensation at k")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="dropout on the encoder's flattened conv features. 0.0 reproduces "
                         "every run collected so far. The encoder is 87%% of the parameters "
                         "and currently carries no regularisation at all, while train loss "
                         "falls 58%% and validation MSE nearly doubles over 30 epochs.")
    ap.add_argument("--weight-decay", type=float, default=FIXED["weight_decay"],
                    help="AdamW weight decay; default 1e-4 is what every collected run used")
    ap.add_argument("--loss", default="mse", choices=("mse", "bin-balanced"),
                    help="mse reproduces every run collected so far; bin-balanced weights "
                         "the three primary bins equally, matching the reported metric")
    ap.add_argument("--fixed-dt", action="store_true", help="feed dt=1 everywhere (question 2)")
    ap.add_argument("--shuffle-frames", action="store_true", help="permute order at TRAIN time")
    ap.add_argument("--per-frame-bug", action="store_true",
                    help="the deliberate augmentation control -- not an option to use")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--name", default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="smoke test only")
    ap.add_argument("--min-free-gb", type=float, default=4.0,
                    help="refuse to start below this much free VRAM; the box is shared. "
                         "Measured footprint is ~2.5 GiB per run, so 4.0 is a 1.6x margin. "
                         "The previous default of 6.0 was a 2.4x margin and refused jobs "
                         "that would have fitted.")
    ap.add_argument("--device", default=None, choices=("cpu", "cuda"))
    args = ap.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        need = args.min_free_gb * 1024 ** 3
        print(f"GPU: {free/1024**3:.1f} GiB free of {total/1024**3:.1f} GiB")
        if free < need:
            # This box is shared with other jobs. Discovering that mid-run costs an epoch and
            # leaves a half-written run directory; discovering it here costs nothing. Never
            # free the memory by killing the other process -- a pkill pattern one character
            # too broad takes down someone's multi-hour job.
            print(f"REFUSING TO START: {free/1024**3:.1f} GiB free, need "
                  f"{args.min_free_gb:.1f}. Something else is using this GPU:", file=sys.stderr)
            subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv"], check=False)
            print("Wait for it, run with --device cpu, or lower --min-free-gb if you are "
                  "certain.", file=sys.stderr)
            raise SystemExit(3)
    set_determinism(args.seed)

    name = args.name or "_".join(
        [args.arm, f"s{args.seed}", f"T{args.T}", f"lr{args.lr:g}", f"k{args.k:g}",
         f"aug-{args.aug}"]
        + (["fixeddt"] if args.fixed_dt else []) + (["shuf"] if args.shuffle_frames else [])
        + (["BUGCTRL"] if args.per_frame_bug else []))
    run = pathlib.Path(args.runs) / name
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)

    data = SteeringData(args.processed, device=dev, pin=True)
    model = build_arm(args.arm, dropout=args.dropout).to(dev)
    brk = model.param_breakdown()
    k_eff = args.k if args.aug == "full" else 0.0
    photo = args.aug != "none"
    translate = args.aug == "full"

    cfg = dict(vars(args), arm_params=brk, device=str(dev), git_sha=git_sha(),
               versions=versions(), fixed=FIXED,
               manifest={k: data.manifest[k] for k in
                         ("n_frames", "splits", "split_buffer", "fps_measured",
                          "target_mean_deg", "target_std_deg", "n_label_dropouts")},
               k_effective=k_eff, photometric=photo, translate=translate)
    (run / "config.json").write_text(json.dumps(cfg, indent=2, default=str) + "\n")
    print(f"[{name}] {brk['total']:,} params "
          f"(encoder {brk['encoder']:,} + recurrent {brk['recurrent']:,} + readout {brk['readout']})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    gen = torch.Generator(device=dev).manual_seed(args.seed + 9999)

    csv_path = run / "metrics.csv"
    cols = ["epoch", "train_loss", "val_macro_skill", "val_global_mae", "val_mse_std",
            "val_curve_mae", "lr", "seconds"]
    with csv_path.open("w", newline="") as fh:
        csv.writer(fh).writerow(cols)

    best, best_epoch, history = -1e9, -1, []
    for epoch in range(args.epochs):
        t0, tot, nb = time.time(), 0.0, 0
        for step, b in enumerate(train_batches(
                data, args.T, FIXED["batch_size"], k_deg_per_px=k_eff,
                epoch_seed=args.seed * 1000 + epoch, per_frame_bug=args.per_frame_bug,
                photometric=photo, translate=translate)):
            f, y, v, dt = b
            if args.shuffle_frames:
                f, y, v, dt = shuffle_within_windows(f, y, v, dt, gen)
            if args.fixed_dt:
                dt = torch.ones_like(dt)
            pred, _ = model(f, dt=dt)
            loss = (masked_mse(pred, y, v) if args.loss == "mse" else
                    masked_bin_balanced_mse(pred, y, v, data.target_mean, data.target_std))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), FIXED["clip_grad_norm"])
            opt.step()
            tot += loss.item(); nb += 1
            if args.max_steps and step + 1 >= args.max_steps:
                break
        sched.step()
        s, mse_std = evaluate_windows(model, data, "val", args.T, FIXED["batch_size"],
                                      args.fixed_dt)
        curve = s["bins"]["curve [15,inf)"]["mae_model"]
        row = [epoch, tot / max(nb, 1), s["macro_skill"], s["global_mae"], mse_std, curve,
               opt.param_groups[0]["lr"], round(time.time() - t0, 1)]
        with csv_path.open("a", newline="") as fh:
            csv.writer(fh).writerow(row)
        history.append(s)
        print(f"  epoch {epoch:2d}  loss {row[1]:.4f}  val skill {s['macro_skill']:+.4f}  "
              f"curve MAE {curve:6.2f}  global MAE {s['global_mae']:6.2f}  {row[-1]}s")

        # SELECTION: validation macro skill. Never global MSE -- it picks the most shrunken
        # checkpoint of every run, undoing the whole point of per-bin metrics. val_mse_std is
        # logged above so the counterfactual is free.
        if s["macro_skill"] > best:
            best, best_epoch = s["macro_skill"], epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": s},
                       run / "checkpoints" / "best.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch, "opt": opt.state_dict()},
                   run / "checkpoints" / "last.pt")
        if epoch - best_epoch >= FIXED["patience"]:
            print(f"  early stop: no improvement for {FIXED['patience']} epochs")
            break

    (run / "final_metrics.json").write_text(json.dumps(
        {"name": name, "arm": args.arm, "seed": args.seed, "best_epoch": best_epoch,
         "best_val_macro_skill": best, "val": history[best_epoch] if history else None,
         "params": brk}, indent=2) + "\n")
    print(f"[{name}] best epoch {best_epoch}, val macro skill {best:+.4f} -> {run}")


if __name__ == "__main__":
    main()
