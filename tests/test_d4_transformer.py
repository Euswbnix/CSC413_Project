"""The supplementary Transformer (docs/preregistration_2026-09-22.md section 9.1), checked on
synthetic input: parameter counts, padding that cannot leak into the prediction, a causal mask,
real (not renumbered) times after frames are dropped, a target frame that is never dropped, and
train/eval running the same maths. Also: the existing arms ignore the new relative-time input."""
import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hdd"))
import d4_run as D  # noqa: E402

DIM = 1536


def model(kind="transformer", seed=0):
    torch.manual_seed(seed)
    return D.Arm(kind, DIM, seed=seed).double()


def batch(lengths, T=10, seed=1):
    """Features, dt, valid and relative times for kept frames packed at the front, as
    forward_masked produces them: irregular real times, the target frame last among the valid."""
    g = torch.Generator().manual_seed(seed)
    B = len(lengths)
    f = torch.randn(B, T, DIM, generator=g, dtype=torch.float64)
    valid = torch.arange(T)[None] < torch.tensor(lengths)[:, None]
    rel = torch.zeros(B, T, dtype=torch.float64)
    for b, n in enumerate(lengths):
        gaps = 0.1 + 0.3 * torch.rand(n - 1, generator=g, dtype=torch.float64)
        times = torch.cat([torch.zeros(1, dtype=torch.float64), gaps.cumsum(0)])
        rel[b, :n] = times - times[-1]
        rel[b, n:] = torch.randn(T - n, generator=g, dtype=torch.float64)       # garbage
    dt = torch.zeros_like(rel)
    dt[:, 1:] = (rel[:, 1:] - rel[:, :-1]).clamp(min=0)
    return f, dt, valid, rel


def test_parameter_counts_match_the_preregistration():
    b = model().blocks()
    assert b["temporal"] == 33_472
    assert b["projection"] == 98_368
    assert b["readout"] == 65
    assert b["temporal"] + b["projection"] + b["readout"] == 131_905
    assert b["temporal_frozen"] == 0


def test_time_code_is_the_registered_sinusoid():
    blk = D.TransformerBlock()
    rel = torch.tensor([[-2.35, -0.4, 0.0]])
    code = blk.time_code(rel)
    p = rel / 0.1
    for j in (0, 1, 7, 31):
        w = 10000 ** (2 * j / 64)
        assert torch.allclose(code[..., 2 * j], torch.sin(p / w), atol=1e-5)
        assert torch.allclose(code[..., 2 * j + 1], torch.cos(p / w), atol=1e-5)
    assert not any(n.startswith("rnn.inv_freq") for n in D.Arm("transformer", DIM).state_dict())


@pytest.mark.parametrize("train_mode", [False, True])
def test_padding_values_cannot_change_the_prediction(train_mode):
    m = model().train(train_mode)
    f, dt, valid, rel = batch([10, 6, 3])
    base = m(f, dt, valid, rel)
    f2, rel2 = f.clone(), rel.clone()
    f2[~valid] = 1e3 * torch.randn(int((~valid).sum()), DIM, dtype=torch.float64)
    rel2[~valid] = -50 + 100 * torch.rand(int((~valid).sum()), dtype=torch.float64)
    assert torch.allclose(m(f2, dt, valid, rel2), base, atol=1e-10)


def test_attention_is_causal():
    m = model().eval()
    f, dt, valid, rel = batch([10])
    x = m.proj(m.norm(f))
    out = m.rnn(x, rel, valid)
    x2 = x.clone()
    x2[:, 6:] += 5.0                                  # change only frames after position 5
    out2 = m.rnn(x2, rel, valid)
    assert torch.allclose(out[:, :6], out2[:, :6], atol=1e-10)
    assert not torch.allclose(out[:, 6:], out2[:, 6:])


def test_train_and_eval_run_the_same_maths():
    m = model()
    f, dt, valid, rel = batch([10, 7, 2])
    assert not torch.backends.mha.get_fastpath_enabled()
    with torch.no_grad():
        a = m.train()(f, dt, valid, rel)
        b = m.eval()(f, dt, valid, rel)
    assert torch.allclose(a, b, atol=1e-12)


def test_times_after_dropping_are_the_real_ones_and_the_target_is_kept():
    torch.manual_seed(0)
    B, T = 64, 30
    step = 0.1 + 0.1 * torch.rand(B, T, dtype=torch.float64)            # irregular sampling
    t = 1000.0 + step.cumsum(1)
    f = torch.randn(B, T, 8)
    seen = {}

    def spy(kf, dt, valid, rel):
        seen.update(kf=kf, dt=dt, valid=valid, rel=rel)
        return torch.zeros(len(kf))

    gen = torch.Generator().manual_seed(7)
    D.forward_masked(spy, f, t, 0.25, gen)
    valid, rel = seen["valid"], seen["rel"].double()
    last = valid.long().cumsum(1).argmax(1)
    assert valid.all(1).logical_not().any()               # frames really were dropped
    assert torch.equal(valid.long().cumsum(1)[torch.arange(B), last], valid.sum(1))
    assert torch.all(rel[torch.arange(B), last].abs() < 1e-6)              # target is last, rel 0
    for b in range(B):
        n = int(valid[b].sum())
        kept_times = rel[b, :n] + t[b, -1]
        # every kept time is one of the window's original timestamps, in order: no renumbering
        idx = [int(torch.argmin((t[b] - v).abs())) for v in kept_times]
        assert torch.allclose(t[b, idx], kept_times, atol=1e-5)
        assert idx == sorted(idx) and idx[-1] == T - 1
    # compacted positions would give integer steps of 0.1 s; real times do not
    assert not torch.allclose(rel[0, :int(valid[0].sum())] % 0.1, torch.zeros(1, dtype=torch.float64), atol=1e-6)


@pytest.mark.parametrize("kind", ["lstm", "cfc"])
def test_existing_arms_ignore_the_relative_times(kind):
    m = model(kind).eval()
    f, dt, valid, rel = batch([10, 4])
    with torch.no_grad():
        assert torch.equal(m(f, dt, valid), m(f, dt, valid, rel))
        assert torch.equal(m(f, dt, valid, rel), m(f, dt, valid, 7 * rel))


def test_transformer_trains_one_step_without_nan():
    m = model().float()
    f, dt, valid, rel = (x.float() if x.is_floating_point() else x for x in batch([10, 5, 1]))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    loss = (m(f, dt, valid, rel) ** 2).mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    opt.step()
    assert math.isfinite(float(loss.detach()))
