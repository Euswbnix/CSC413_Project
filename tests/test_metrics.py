"""Metrics, with a MUST-FIRE case for every property that matters.

Written this way on purpose. In the previous round almost every bug was in checking code
rather than in the code being checked -- a revisit audit that could not answer its own
question, a stripe detector that fired twice on noise -- and the common cause was testing
only that each check stayed quiet on good input. Every assertion below therefore has a
partner that constructs input on which it MUST fail.
"""

import numpy as np
import pytest

from metrics import (BIN_NAMES, bin_masks, count_turn_events, false_alarm_rate,
                     mae_by_window_position, macro_skill, per_bin, persistence, pearson_r,
                     summary)


# ------------------------------------------------------------------ turn events ----

def test_turn_events_merge_short_gaps_and_separate_long_ones():
    m = np.zeros(200, bool)
    m[10:20] = True
    m[25:35] = True          # 5-frame gap -> same event
    m[100:110] = True        # 65-frame gap -> separate
    assert count_turn_events(m) == 2


def test_turn_events_must_not_report_one_corner_as_many_samples():
    """MUST-FIRE: 200 contiguous frames are ONE event. If this ever returns ~200 the count
    has become a frame count and every table's support column is wrong."""
    assert count_turn_events(np.ones(200, bool)) == 1
    assert count_turn_events(np.zeros(200, bool)) == 0


# ---------------------------------------------------------------------- binning ----

def test_bins_are_assigned_by_ground_truth_not_prediction():
    """MUST-FIRE: a constant-zero predictor must still populate the curve bin. Binning by
    prediction would leave it empty and hide the failure instead of scoring it."""
    true = np.array([0.0, 8.0, 30.0, 80.0])
    names = [n for n, m in bin_masks(true) if m.any()]
    assert "curve [15,inf)" in names
    rows = per_bin(np.zeros(4), true)
    assert rows["curve [15,inf)"]["n_frames"] == 2
    assert rows["curve [15,inf)"]["mae_model"] == pytest.approx(55.0)


def test_diagnostic_bin_is_reported_but_is_not_a_primary_bin():
    true = np.array([0.0, 8.0, 30.0, 80.0])
    rows = per_bin(np.zeros(4), true)
    diag = [k for k in rows if k.startswith("(diagnostic)")]
    assert len(diag) == 1 and rows[diag[0]]["n_frames"] == 1
    assert diag[0] not in BIN_NAMES
    # and it must NOT enter the headline scalar: 2 test events is not a measurement
    assert sum(1 for k in rows if k in BIN_NAMES) == 3


# -------------------------------------------------------------- the validity mask ----

def test_invalid_frames_are_excluded_everywhere():
    true = np.array([0.0, 0.0, 0.0, 80.0])
    valid = np.array([True, True, False, True])
    rows = per_bin(np.zeros(4), true, valid)
    assert rows["straight [0,5)"]["n_frames"] == 2


def test_the_mask_demonstrably_changes_the_answer():
    """MUST-FIRE: a dropout is a label of exactly 0 recorded mid-corner. Included, it lands
    in the straight bin and flatters predict-0 there. If masking ever becomes a no-op this
    fails, which is the point -- the previous round's bugs all passed their own tests."""
    true = np.concatenate([np.full(50, 80.0), [0.0], np.full(50, 80.0)])   # a dropout at 50
    valid = np.ones(101, bool); valid[50] = False
    pred = np.full(101, 80.0)
    with_bad = per_bin(pred, true)
    without = per_bin(pred, true, valid)
    assert with_bad["straight [0,5)"]["n_frames"] == 1
    assert without["straight [0,5)"]["n_frames"] == 0
    assert with_bad["straight [0,5)"]["mae_model"] == pytest.approx(80.0)
    assert np.isnan(without["straight [0,5)"]["mae_model"])


# ------------------------------------------------------------------ macro skill ----

def test_predict_zero_scores_exactly_zero():
    """The property the headline scalar is chosen for: the baseline is pinned at 0, so the
    score cannot be won by shrinking predictions toward the mean."""
    rng = np.random.default_rng(0)
    true = rng.normal(0, 30, 4000)
    assert macro_skill(np.zeros_like(true), true) == pytest.approx(0.0, abs=1e-12)


def test_a_shrunken_predictor_cannot_win():
    """MUST-FIRE: shrinkage is the failure mode binning exists to prevent. A predictor that
    halves every target must NOT outscore one that is genuinely closer."""
    rng = np.random.default_rng(1)
    true = rng.normal(0, 30, 4000)
    shrunk = 0.5 * true
    honest = true + rng.normal(0, 3, 4000)
    assert macro_skill(honest, true) > macro_skill(shrunk, true)
    assert macro_skill(true, true) == pytest.approx(1.0)


def test_macro_skill_ignores_the_diagnostic_bin():
    """Two independent turn events must not control a sixth of the headline scalar."""
    true = np.concatenate([np.full(300, 1.0), np.full(300, 8.0), np.full(300, 20.0),
                           np.full(4, 200.0)])
    pred = true.copy()
    pred[-4:] = 0.0                     # catastrophic, but only in the diagnostic bin
    assert macro_skill(pred, true) > 0.99


# -------------------------------------------------------------- other quantities ----

def test_persistence_uses_the_previous_true_angle():
    true = np.array([1.0, 2.0, 5.0])
    assert list(persistence(true)) == [1.0, 1.0, 2.0]


def test_pearson_and_false_alarm():
    true = np.array([0.0, 1.0, 2.0, 3.0])
    assert pearson_r(2 * true, true) == pytest.approx(1.0)
    assert pearson_r(-true, true) == pytest.approx(-1.0)
    # predict-0 aces the false-alarm rate by construction; say so rather than celebrate it
    assert false_alarm_rate(np.zeros(4), true) == 0.0
    assert false_alarm_rate(np.full(4, 50.0), true) == 1.0


# ------------------------------------------- per-position curve and its self-test ----

def test_a_length_invariant_arm_gives_a_flat_position_curve():
    """The headline figure's own self-test. A non-recurrent arm has no state, so its error
    cannot depend on position within the window; a sloped line would mean the position
    accounting is broken, not that the CNN grew memory."""
    rng = np.random.default_rng(2)
    T, N = 16, 400
    true = rng.normal(0, 20, (N, T))
    pred = true + rng.normal(0, 4, (N, T))          # position-independent error
    curve = mae_by_window_position(pred, true, np.ones((N, T), bool), T)
    assert curve.shape == (T,)
    assert curve.std() < 0.35, f"flat curve expected, got spread {curve.std():.3f}"


def test_the_position_curve_detects_a_warm_up():
    """MUST-FIRE: a recurrent model should improve with context. If the aggregation could
    not see that, the headline figure would be decorative."""
    rng = np.random.default_rng(3)
    T, N = 16, 400
    true = rng.normal(0, 20, (N, T))
    noise = np.linspace(12.0, 1.0, T)[None, :]      # error shrinking with position
    pred = true + rng.normal(0, 1, (N, T)) * noise
    curve = mae_by_window_position(pred, true, np.ones((N, T), bool), T)
    assert curve[0] > 3 * curve[-1]
    assert np.all(np.diff(curve) < 1.0)


def test_summary_is_serialisable_and_complete():
    import json
    rng = np.random.default_rng(4)
    true = rng.normal(0, 25, 500)
    s = summary(true + rng.normal(0, 5, 500), true)
    json.dumps(s)
    assert set(s) >= {"macro_skill", "global_mae", "pearson_r", "false_alarm_rate", "bins"}
