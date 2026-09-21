"""LayerNorm option, encoder learning rate, run naming and the health probes.

Every behaviour that must stay OFF by default has a partner check that it actually turns ON,
because an option that silently does nothing would report "no effect" for an experiment that
never ran.
"""
import pathlib
import sys
import types

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import diagnostics
import train
from models.interface import build_arm

X = torch.randn(2, 4, 3, 66, 200)
DT = torch.ones(2, 4)


def args(**kw):
    base = dict(name=None, arm="cfc", seed=0, T=16, lr=1e-3, k=0.1, aug="full", fixed_dt=False,
                shuffle_frames=False, per_frame_bug=False, loss="mse", dropout=0.0,
                weight_decay=1e-4, feature_norm=False, encoder_lr=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


# ------------------------------------------------------------------------- LayerNorm

def test_default_state_dict_is_unchanged():
    torch.manual_seed(0); a = build_arm("cfc")
    torch.manual_seed(0); b = build_arm("cfc", feature_norm=False)
    assert list(a.state_dict()) == list(b.state_dict())
    assert not any("feature_norm" in k for k in a.state_dict())
    a.eval(); b.eval()
    assert torch.equal(a(X, dt=DT)[0], b(X, dt=DT)[0])


def test_feature_norm_normalises_and_adds_no_parameters():
    """Checked against the exact LayerNorm formula, not "variance ~ 1": at initialisation the
    raw features have variance ~1e-3, so eps = 1e-5 is not negligible and the normalised
    variance is v/(v+eps) ~ 0.99. (That is also a reminder that LayerNorm changes the start of
    training: it lifts features from RMS ~0.03 to ~1 before any step.)"""
    for arm, n in (("cfc", 193_141), ("cnn_2frame", 170_077)):
        torch.manual_seed(0); on = build_arm(arm, feature_norm=True).eval()
        torch.manual_seed(0); off = build_arm(arm).eval()     # same weights: LN draws no RNG
        assert sum(q.numel() for q in on.parameters()) == n
        raw = off.encoder(X)
        want = torch.nn.functional.layer_norm(raw, (raw.shape[-1],))
        assert torch.allclose(on.encoder(X), want, atol=1e-6)
        assert torch.allclose(on.encoder(X).mean(-1), torch.zeros(2, 4), atol=1e-5)
        assert raw.var(-1, unbiased=False).max() < 0.1          # MUST differ without it


def test_architecture_mismatch_fails_loudly_both_ways():
    on, off = build_arm("cfc", feature_norm=True), build_arm("cfc")
    with pytest.raises(RuntimeError):
        off.load_state_dict(on.state_dict())
    with pytest.raises(RuntimeError):
        on.load_state_dict(off.state_dict())


# ------------------------------------------------------------- optimiser and naming

def test_default_optimizer_is_one_group():
    m = build_arm("cfc")
    opt = train.make_optimizer(m, args())
    assert len(opt.param_groups) == 1 and opt.param_groups[0]["lr"] == 1e-3


def test_encoder_lr_splits_exactly_the_encoder():
    m = build_arm("cfc")
    opt = train.make_optimizer(m, args(encoder_lr=1e-4))
    g_enc, g_rest = opt.param_groups
    assert g_enc["lr"] == 1e-4 and g_rest["lr"] == 1e-3
    assert {id(q) for q in g_enc["params"]} == {id(q) for q in m.encoder.parameters()}
    assert (sum(q.numel() for q in g_enc["params"]) + sum(q.numel() for q in g_rest["params"])
            == sum(q.numel() for q in m.parameters()))


def test_default_run_name_is_unchanged():
    assert train.run_name(args()) == "cfc_s0_T16_lr0.001_k0.1_aug-full"


def test_non_default_options_reach_the_run_name():
    names = {train.run_name(args(**kw)) for kw in (
        {}, {"loss": "bin-balanced"}, {"dropout": 0.3}, {"weight_decay": 1e-3},
        {"feature_norm": True}, {"encoder_lr": 1e-4}, {"feature_norm": True, "encoder_lr": 1e-4})}
    assert len(names) == 7                                    # all distinct: no overwrite
    assert train.run_name(args(feature_norm=True, encoder_lr=1e-4)).endswith("_fn_elr0.0001")


# -------------------------------------------------------------------- health probes

def test_gate_stats_at_init_and_when_forced_to_saturate():
    frames = torch.rand(8, 1, 3, 66, 200)
    for arm in ("cfc", "lstm"):
        torch.manual_seed(0)
        m = build_arm(arm)
        g = diagnostics.gate_stats(m, frames)
        assert g["sigmoid_saturated_fraction"] < 0.05       # measured 0.0 at init in the review
        with torch.no_grad():
            m.encoder.fc.weight.mul_(1e4); m.encoder.fc.bias.fill_(50.0)
        g = diagnostics.gate_stats(m, frames)
        assert g["feature_rms"] > 10
        assert g["sigmoid_saturated_fraction"] > 0.5        # MUST fire on blown-up features
    assert "sigmoid_saturated_fraction" not in diagnostics.gate_stats(build_arm("cnn_linear"), frames)


def test_gate_stats_restores_training_mode():
    m = build_arm("cfc", dropout=0.5).train()
    diagnostics.gate_stats(m, torch.rand(2, 1, 3, 66, 200))
    assert m.training


def test_group_grad_norms_partition_the_total():
    m = build_arm("cfc")
    out, _ = m(X, dt=DT)
    out.square().mean().backward()
    g = diagnostics.group_grad_norms(m)
    total = torch.stack([q.grad.norm() for q in m.parameters() if q.grad is not None]).norm()
    parts = (g["grad_encoder"] ** 2 + g["grad_recurrent"] ** 2 + g["grad_readout"] ** 2) ** 0.5
    assert abs(parts - float(total)) < 1e-4 * max(1.0, float(total))
