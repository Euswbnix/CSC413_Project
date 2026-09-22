"""The lag estimator in hdd/sync_check.py must recover a lag it is given, on synthetic signals.

This is the regression test for the bug found on 2026-09-22: correlating the ACCUMULATED heading,
and normalising np.correlate by the full length instead of the overlap, both pull the estimate
toward zero, so a real offset reads as "aligned".
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "hdd"))
import sync_check as S


def signals(n=1800, seed=0):
    """A yaw-rate-like signal and the image motion it would produce, both per frame."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    yaw = sum(np.sin(2 * np.pi * t / p + rng.random() * 6.3) * a
              for p, a in ((90, 1.0), (240, 2.0), (600, 3.0)))
    return yaw + rng.normal(0, 0.05, n)


@pytest.mark.parametrize("true_lag", [-20, -7, -3, 0, 3, 7, 20])
def test_recovers_injected_lag(true_lag):
    e = signals()
    f = np.roll(e, true_lag)                      # the image signal happens `true_lag` later
    cut = abs(true_lag) + 1
    k, corr, sharp = S.ncc_lag(f[cut:-cut], e[cut:-cut], max_lag=40)
    assert k == true_lag, f"recovered {k}, expected {true_lag}"
    assert abs(corr) > 0.9 and sharp > 0.1


def test_quantisation_does_not_move_the_peak():
    """Rounding the image signal to whole columns lowers the correlation but not the lag."""
    e = signals(seed=1)
    f = np.round(np.roll(e, -4))
    k, corr, _ = S.ncc_lag(f[5:-5], e[5:-5], max_lag=40)
    assert k == -4 and abs(corr) > 0.5


def test_taper_normalisation_pulls_the_estimate_toward_zero():
    """Why the correlation is normalised by the overlap: dividing by the full length imposes a
    triangular taper, which is the bug we shipped on 2026-09-22 (with accumulated signals it was
    enough to pin every estimate at zero)."""
    e = signals(seed=2)
    lag = -30
    f = np.roll(e, lag)
    a, b = f[31:-31], e[31:-31]
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    c = np.correlate(a, b, mode="full")
    mid, max_lag = len(b) - 1, 40
    ks = np.arange(-max_lag, max_lag + 1)
    seg = c[mid - max_lag:mid + max_lag + 1]
    tapered = ks[np.argmax(np.abs(seg / len(a)))]          # what the old code did
    overlap = ks[np.argmax(np.abs(seg / (len(a) - np.abs(ks))))]
    assert abs(overlap - lag) <= 1                         # edges cost at most a sample
    assert abs(tapered) <= abs(overlap)                    # the taper can only pull it inward


def test_low_frequency_signal_still_recovers():
    """A slow, smooth signal (what accumulation produces) is the hard case for a lag estimate."""
    t = np.arange(1800)
    e = np.sin(2 * np.pi * t / 900) + 0.3 * np.sin(2 * np.pi * t / 300)
    k, corr, _ = S.ncc_lag(np.roll(e, -12)[13:-13], e[13:-13], max_lag=40)
    assert k == -12 and abs(corr) > 0.99
