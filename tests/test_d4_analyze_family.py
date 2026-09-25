"""hdd/d4_analyze.py: Holm, the bootstrap p-values, and the registered comparison family
(docs/preregistration_2026-09-22.md sections 5 and 9.4), on synthetic predictions."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hdd"))
import d4_analyze as A  # noqa: E402


def test_holm_matches_a_worked_example():
    adj = A.holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(A.holm([0.2, 0.9]), [0.4, 0.9])
    assert np.all(A.holm(np.random.default_rng(0).random(20)) <= 1.0)


def test_p_values_agree_with_the_interval_rules():
    rng = np.random.default_rng(1)
    for mean, sd in [(3.0, 0.5), (0.0, 0.5), (0.1, 0.05), (-0.6, 0.02), (0.4, 0.3)]:
        d = rng.normal(mean, sd, 20000)
        lo, hi = np.percentile(d, [2.5, 97.5])
        p_diff, p_equiv = A.p_values(d, 0.5)
        excludes_zero, inside_delta = lo > 0 or hi < 0, -0.5 < lo and hi < 0.5
        if abs(np.mean(d <= 0) - 0.025) > 0.005:          # away from the knife edge
            assert (p_diff < 0.05) == excludes_zero, (mean, sd, p_diff, lo, hi)
        if min(abs(np.mean(d >= 0.5) - 0.025), abs(np.mean(d <= -0.5) - 0.025)) > 0.005:
            assert (p_equiv < 0.05) == inside_delta, (mean, sd, p_equiv, lo, hi)
    assert A.p_values(np.full(99, 5.0), None) == (0.02, None)       # never exactly zero


def _fake_predictions(tmp, arms, seeds=3, n=600):
    rng = np.random.default_rng(2)
    y = rng.normal(0, 15, n)
    cluster = np.repeat(np.arange(6), n // 6)
    session = np.array([f"2017030{c}1200" for c in cluster])
    offset = dict(lstm=0.0, cfc=0.2, ltc=3.0)
    for arm in arms:
        for s in range(seeds):
            preds = {}
            for keep, draws in ((1.0, 1), (0.5, 5), (0.25, 5)):
                for m in range(draws):
                    noise = rng.normal(0, 4 + (1 - keep), n)
                    preds[f"k{keep:g}_m{m}"] = y + noise + np.sign(noise) * offset[arm]
            np.savez(os.path.join(tmp, f"pred_{arm}_s{s}_val.npz"), y=y, cluster=cluster,
                     session=session, **preds)


def test_family_has_one_primary_and_eleven_secondary_tests(tmp_path):
    _fake_predictions(str(tmp_path), ["lstm", "cfc", "ltc"])
    res = A.family(str(tmp_path), "val", n_boot=5000)
    assert res["primary"]["test"] == "cfc-lstm keep_0.25"
    names = [t["test"] for t in res["secondary"]]
    assert len(names) == res["secondary_m"] == 11 and "cfc-lstm keep_0.25" not in names
    assert {"cfc-lstm H-B", "ltc-lstm H-B", "ltc-cfc H-B"} <= set(names)
    for t in res["secondary"]:
        assert t["p_diff_holm"] >= t["p_diff"]
        if t["test"].endswith("H-B"):
            assert t["p_equiv"] is None and t["verdict_holm"] != "practically equivalent"
    ltc = [t for t in res["secondary"] if t["test"].startswith("ltc-lstm keep")]
    assert all(t["verdict_holm"] == "different" for t in ltc)   # 3 deg apart on this fake data


def test_family_reports_a_missing_pair_instead_of_dropping_it(tmp_path):
    _fake_predictions(str(tmp_path), ["lstm", "cfc"])
    res = A.family(str(tmp_path), "val", n_boot=5000)
    assert res["missing_pairs"] == ["ltc-lstm", "ltc-cfc"]
    assert res["secondary_m"] == 3


def test_transformer_never_enters_the_family(tmp_path):
    _fake_predictions(str(tmp_path), ["lstm", "cfc", "ltc"])
    import shutil
    for f in os.listdir(tmp_path):
        if f.startswith("pred_lstm"):
            shutil.copy(os.path.join(tmp_path, f), os.path.join(tmp_path, f.replace("lstm", "transformer")))
    res = A.family(str(tmp_path), "val", n_boot=5000)
    assert not any("transformer" in t["test"] for t in res["secondary"])


def test_family_refuses_too_few_bootstrap_draws(tmp_path):
    import pytest
    _fake_predictions(str(tmp_path), ["lstm", "cfc", "ltc"])
    with pytest.raises(SystemExit):
        A.family(str(tmp_path), "val", n_boot=300)
