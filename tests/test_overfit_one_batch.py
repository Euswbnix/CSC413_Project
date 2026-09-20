"""REVIEW B1's hardened gate, promoted into the suite so it re-runs after every refactor.

The batch MUST contain a |angle| > 40 deg frame and the criterion is per-frame error on
THAT frame -- a plain "loss goes down" gate passes while the model is incapable of
emitting a sharp turn at all.

This is the only assertion in the suite that fails on the label-scale trap: the recurrent
state is strictly bounded to (-1,1), so fitting raw degrees against a +-100 tail reaches
only ~87 deg after thousands of steps, which looks exactly like "the recurrent cell cannot
handle sharp turns" -- a fabricated architectural finding. Standardised targets reach
~99.8. test_raw_degree_targets_underfit_the_sharp_turn documents that.

Synthetic frames on purpose: this gates the OPTIMISER and OUTPUT-RANGE paths, which need
no dataset, so it stays off the critical path in week 1.
"""

import pytest
import torch

from models.interface import build_arm

pytestmark = pytest.mark.slow

B, T, STEPS = 2, 8, 400


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    frames = torch.rand(B, T, 3, 66, 200, generator=g)
    angles = torch.randn(B, T, 1, generator=g) * 8.0
    angles[0, 3, 0] = 78.0          # the frame the gate is actually about
    angles[1, 6, 0] = -52.0
    return frames, angles


def fit(standardise, steps=STEPS, lr=3e-3):
    torch.manual_seed(0)
    m = build_arm("cfc")
    frames, angles = batch()
    mu, sd = (angles.mean(), angles.std()) if standardise else (0.0, 1.0)
    target = (angles - mu) / sd
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        pred, _ = m(frames, dt=torch.ones(B, T))
        loss = (pred - target).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        pred_deg = m(frames, dt=torch.ones(B, T))[0] * sd + mu
    return (pred_deg - angles).abs()


def test_overfits_one_batch_including_the_sharp_turn():
    err = fit(standardise=True)
    assert err[0, 3, 0].item() < 1.0, f"sharp-turn frame error {err[0, 3, 0].item():.2f} deg"
    assert err.max().item() < 2.0, f"worst per-frame error {err.max().item():.2f} deg"


def test_raw_degree_targets_underfit_the_sharp_turn():
    """Documents the trap rather than merely avoiding it: the SAME model, SAME steps, on
    un-standardised degree targets, cannot reach the 78 deg frame. If this ever starts
    passing, the bound on the recurrent state has changed and MODEL.md is stale."""
    err = fit(standardise=False)
    assert err[0, 3, 0].item() > 2.0, (
        "raw-degree targets no longer underfit -- re-derive the B1 argument before "
        "relaxing the standardisation requirement"
    )
