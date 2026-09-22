"""Training-health probes: encoder feature scale and first-step gate saturation.

train.py logs these every epoch; `python diagnostics.py RUN_ROOT --processed DIR` probes saved
checkpoints after the fact (initial weights vs best checkpoint, see docs/diagnosis_2026-09-20.md
section 18). Both use the same definitions, so the two measure the same thing:

* inputs: `n` validation frames at np.linspace(lo, hi - 1, n).astype(int), centre-cropped to
  the model width, scaled to [0, 1];
* zero initial state, dt = 1, first step only;
* a sigmoid is saturated outside (0.01, 0.99), a tanh at |x| > 0.99.

First step only is a real limitation: it says nothing about saturation deep in a rollout, where
the state is no longer zero. It is kept because it is cheap enough to log every epoch and it is
what docs/diagnosis_2026-09-20.md section 18.3 reports (a clean re-run; the review's own figures
came from two approximate image patches and are not directly comparable).
"""
import argparse
import json
import pathlib
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIG_LO, SIG_HI, TANH_SAT = 0.01, 0.99, 0.99


def probe_frames(data, n=128, split="val"):
    """(n, 1, 3, H, model_w) float tensor on data.device, matching the review's probe inputs."""
    lo, hi = data.manifest["splits"][split]
    idx = torch.from_numpy(np.linspace(lo, hi - 1, n).astype(int))
    f = data.frames[idx]                                          # (n, H, W_stored, 3) uint8
    off = (f.shape[2] - data.model_w) // 2
    f = f[:, :, off:off + data.model_w].to(data.device).float().div_(255.0)
    return f.permute(0, 3, 1, 2)[:, None].contiguous()


@torch.no_grad()
def gate_stats(model, frames):
    """Feature RMS and first-step gate saturation. Keys absent for arms without gates."""
    was_training = model.training
    model.eval()
    try:
        z = model.encoder(frames)[:, 0]
        out = {"feature_rms": float(z.square().mean().sqrt())}
        rnn = getattr(model.rnn, "rnn", None)
        if isinstance(rnn, nn.LSTM):
            H = rnn.hidden_size
            # Zero h and c: W_hh h vanishes. PyTorch gate order is i, f, g, o.
            a = F.linear(z, rnn.weight_ih_l0, rnn.bias_ih_l0) + rnn.bias_hh_l0
            g = [a[:, j * H:(j + 1) * H] for j in range(4)]
            tanh = g[2].tanh()
            sig = torch.cat([g[j].sigmoid() for j in (0, 1, 3)], -1)
        elif rnn is not None and hasattr(rnn, "cell"):
            c = rnn.cell
            x = torch.cat([z, z.new_zeros(z.shape[0], c.hidden_size)], -1)
            if c.backbone is not None:
                x = c.backbone(x)
            tanh = torch.cat([c._aff(c.h_head, x, True).tanh(),
                              c._aff(c.g_head, x, True).tanh()], -1)
            sig = (c._aff(c.f_head, x, c.mask_gate) * 1.0      # dt = 1
                   + c._aff(c.o_head, x, c.mask_gate)).sigmoid()
        else:
            return out
        out["sigmoid_saturated_fraction"] = float(((sig < SIG_LO) | (sig > SIG_HI)).float().mean())
        out["tanh_saturated_fraction"] = float((tanh.abs() > TANH_SAT).float().mean())
        return out
    finally:
        model.train(was_training)


def group_grad_norms(model):
    """L2 norm of the current gradients of encoder, recurrent block and readout (pre-clip)."""
    def norm(m):
        g = [p.grad.detach().float().norm() for p in m.parameters() if p.grad is not None]
        return float(torch.stack(g).norm()) if g else 0.0
    return {"grad_encoder": norm(model.encoder), "grad_recurrent": norm(model.rnn),
            "grad_readout": norm(model.readout)}


def frames_from_dir(processed, n=128, split="val", model_w=200):
    """probe_frames() for a processed directory, without building the training dataset."""
    d = pathlib.Path(processed)
    lo, hi = json.loads((d / "manifest.json").read_text())["splits"][split]
    a = np.load(d / "frames.npy", mmap_mode="r")
    idx = np.linspace(lo, hi - 1, n).astype(int)
    off = (a.shape[2] - model_w) // 2
    x = np.asarray(a[idx, :, off:off + model_w], dtype=np.float32) / 255.0
    return torch.from_numpy(x).permute(0, 3, 1, 2)[:, None].contiguous()


def probe_checkpoints(root, frames):
    """For every CfC/LSTM run under `root`: gate_stats at the seed's initial weights and at the
    best checkpoint, plus the saved test predictions' std ratio. Read-only."""
    from models.interface import build_arm
    records = []
    for cp in sorted(pathlib.Path(root).glob("*/config.json")):
        cfg = json.loads(cp.read_text())
        if cfg["arm"] not in ("cfc", "lstm") or not (cp.parent / "checkpoints/best.pt").exists():
            continue
        torch.manual_seed(cfg["seed"])            # train.py seeds before building the model
        model = build_arm(cfg["arm"], dropout=cfg.get("dropout", 0.0),
                          feature_norm=cfg.get("feature_norm", False))
        initial = gate_stats(model, frames)
        ck = torch.load(cp.parent / "checkpoints/best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        rec = dict(run=cp.parent.name, arm=cfg["arm"], seed=cfg["seed"], epoch=ck["epoch"],
                   initial=initial, trained=gate_stats(model, frames))
        pt = cp.parent / "predictions_test.npz"
        if pt.exists():
            with np.load(pt) as z:
                p, y, v = z["pred"].astype(float), z["true"].astype(float), z["valid"].astype(bool)
            rec["test_std_ratio"] = float(p[v].std() / y[v].std())
        records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser(description="Probe feature scale and gate saturation of saved runs.")
    ap.add_argument("root", help="directory of run folders, e.g. runs_power")
    ap.add_argument("--processed", required=True, help="processed data dir (manifest.json, frames.npy)")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", type=pathlib.Path, help="optional JSON output (results/ is git-ignored)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    recs = probe_checkpoints(a.root, frames_from_dir(a.processed, a.n))
    if not recs:
        sys.exit(f"no cfc/lstm run with checkpoints/best.pt directly under {a.root}")
    summary = []
    for config in sorted({re.sub(r"_s\d+_", "_", r["run"]) for r in recs}):   # one line per configuration
        rs = [r for r in recs if re.sub(r"_s\d+_", "_", r["run"]) == config]
        s = dict(config=config, arm=rs[0]["arm"], n=len(rs))
        for stage in ("initial", "trained"):
            s[stage] = {k: float(np.median([r[stage][k] for r in rs])) for k in rs[0][stage]}
        s["sigmoid_saturated_over_90pct"] = sum(r["trained"].get("sigmoid_saturated_fraction", 0) > 0.9 for r in rs)
        summary.append(s)
    print(json.dumps(summary, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(scope=f"{a.n} {a.processed} validation frames; zero state, dt=1",
                                         summary=summary, records=recs), indent=2) + "\n")


if __name__ == "__main__":
    main()
