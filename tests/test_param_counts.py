"""Layer 3 -- the parameter table, asserted rather than transcribed.

scripts/param_table.py emits the README's tables from the SAME formula functions these
tests assert against, so the table is generated output. `make test && make params` is the
answer to "how do I check this?".
"""

import pytest
import torch
import torch.nn as nn

from models.cfc import CfCCell
from models.encoder import PilotNetEncoder, TwoFrameEncoder, FLATTEN_WIDTH
from models.interface import (build_arm, cfc_params, lstm_params, gru_params, mlp_params,
                              matched_width, HIDDEN)

I = 32
WIDTHS = [16, 32, 64, 128]


@pytest.mark.parametrize("H", WIDTHS)
def test_cfc_formula_matches_named_parameters(H):
    assert sum(p.numel() for p in CfCCell(I, H).parameters()) == cfc_params(I, H)


@pytest.mark.parametrize("H", WIDTHS)
def test_lstm_minus_cfc_is_exactly_four_H(H):
    """The residual is nothing but torch's redundant second bias vector (cuDNN's RNN
    convention), and it is IN THE BASELINE'S FAVOUR -- we did not shrink our opponent.
    Measured: 3136/3200, 8320/8448, 24832/25088, 82432/82944."""
    n_cfc = sum(p.numel() for p in CfCCell(I, H).parameters())
    n_lstm = sum(p.numel() for p in nn.LSTM(I, H, batch_first=True).parameters())
    assert n_lstm == lstm_params(I, H)
    assert n_lstm - n_cfc == 4 * H


def test_headline_configuration_numbers():
    """The exact numbers that go in the README. If one of these moves, the model figure
    and the parameter table are stale."""
    assert cfc_params(32, 64) == 24_832
    assert lstm_params(32, 64) == 25_088
    enc = PilotNetEncoder()
    assert sum(p.numel() for p in enc.parameters()) == 168_244
    assert sum(p.numel() for p in TwoFrameEncoder().parameters()) == 170_044
    m = build_arm("cfc")
    b = m.param_breakdown()
    assert b == {"encoder": 168_244, "recurrent": 24_832, "readout": 65, "total": 193_141}
    share = b["recurrent"] / b["total"]
    assert 0.128 < share < 0.130, f"recurrent share {share:.3%}, expected 12.9%"


def test_no_frozen_parameters_in_any_arm():
    """The reference library reports 6,898 for a CfC whose trainable count is 5,544 -- a
    24.4% overcount -- because it registers its sparsity mask as an nn.Parameter with
    requires_grad=False. We use register_buffer. PROJECT.md section 3's snippet would have
    printed the wrong number straight into the README."""
    for name in ("cfc", "lstm", "gru", "cnn_mlp", "cnn_linear", "cnn_2frame"):
        m = build_arm(name)
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        assert total == trainable, f"{name} has frozen parameters counted as parameters"


def test_mask_is_a_buffer_by_identity_and_is_in_state_dict():
    M = (torch.rand(8, 12) > 0.5).float()
    c = CfCCell(4, 8, mask=M)
    assert not any(p is c.mask for p in c.parameters())
    assert "mask" in c.state_dict()
    assert sum(p.numel() for p in c.parameters()) == cfc_params(4, 8)


def test_controls_are_matched_to_the_cfc_within_tolerance():
    target = cfc_params(I, HIDDEN)
    counts = {n: sum(p.numel() for p in build_arm(n).rnn.parameters()) for n in
              ("cfc", "lstm", "gru")}
    counts["cnn_mlp_head"] = sum(p.numel() for p in build_arm("cnn_mlp").readout.parameters())
    for name in ("lstm", "gru", "cnn_mlp_head"):
        ratio = counts[name] / target
        assert 0.97 <= ratio <= 1.03, f"{name} is {ratio:.3f}x the CfC ({counts[name]} vs {target})"


def test_gru_and_mlp_matched_widths_are_what_we_publish():
    target = cfc_params(I, HIDDEN)
    assert matched_width(target, gru_params, I) == 76
    assert gru_params(I, 76) == 25_080
    assert matched_width(target, mlp_params, I) == 730
    assert mlp_params(I, 730) == 24_821


def test_cnn_linear_is_the_lower_bracket():
    """33 parameters. Together with cnn_mlp (24,821, matched) the two non-recurrent heads
    bracket the recurrent arms from below and at matched capacity, so the cost of deleting
    TEMPORAL STATE cannot be confused with the cost of deleting HEAD CAPACITY."""
    assert sum(p.numel() for p in build_arm("cnn_linear").readout.parameters()) == 33
