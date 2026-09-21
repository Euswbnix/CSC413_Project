#!/usr/bin/env python3
"""The encoder's dropout must be OFF by default and must actually fire when asked.

Both halves matter. Off by default is what keeps the 115 collected runs reproducible. But a
dropout that is silently inert would be worse than none: the regularisation sweep would
report that regularisation does not help, when in fact it never ran. `TwoFrameEncoder`
overrides `forward` and so needs its own wiring -- an earlier draft added dropout to the
base class and that subclass quietly bypassed it.
"""
import pathlib, sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from models.interface import build_arm

ARMS = ("cfc", "cnn_2frame")
X = torch.randn(2, 4, 3, 66, 200)
DT = torch.ones(2, 4)


def test_default_is_deterministic_and_unchanged():
    for arm in ARMS:
        m = build_arm(arm).train()
        a, _ = m(X, dt=DT)
        b, _ = m(X, dt=DT)
        assert torch.equal(a, b), f"{arm}: default must be dropout-free"


def test_dropout_actually_fires_in_train_mode():
    for arm in ARMS:
        m = build_arm(arm, dropout=0.5).train()
        a, _ = m(X, dt=DT)
        b, _ = m(X, dt=DT)
        assert not torch.equal(a, b), (
            f"{arm}: two train-mode passes are identical at dropout=0.5 -- the dropout is "
            f"not wired into this encoder's forward")


def test_eval_mode_is_deterministic_again():
    for arm in ARMS:
        m = build_arm(arm, dropout=0.5).eval()
        a, _ = m(X, dt=DT)
        b, _ = m(X, dt=DT)
        assert torch.equal(a, b), f"{arm}: eval mode must disable dropout"


def test_parameter_count_is_untouched():
    for arm, expected in (("cfc", 193_141), ("cnn_2frame", 170_077)):
        n0 = sum(p.numel() for p in build_arm(arm).parameters())
        n1 = sum(p.numel() for p in build_arm(arm, dropout=0.5).parameters())
        assert n0 == n1 == expected, f"{arm}: {n0} / {n1}, expected {expected}"
