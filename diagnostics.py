"""Training-health probes: encoder feature scale and first-step gate saturation.

The definitions follow scripts/probe_saved_gates.py (docs/training_review_2026-09-21.md,
section 2) exactly, so the per-epoch log written by train.py and the post-hoc probe of a saved
checkpoint measure the same thing:

* inputs: `n` validation frames at np.linspace(lo, hi - 1, n).astype(int), centre-cropped to
  the model width, scaled to [0, 1];
* zero initial state, dt = 1, first step only;
* a sigmoid is saturated outside (0.01, 0.99), a tanh at |x| > 0.99.

First step only is a real limitation: it says nothing about saturation deep in a rollout, where
the state is no longer zero. It is kept because it is cheap enough to log every epoch and it is
what the review measured, so the two can be compared directly.
"""
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
