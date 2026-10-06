"""hdd/anticip_analyze.py on synthetic predictions with a known structure
(docs/plan_2026-10-05_steering_anticipation.md, questions Q1-Q4)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hdd"))
import anticip_analyze as Z  # noqa: E402

# mean |error| added on top of the noise, by arm and horizon (0, 0.5, 1, 2 s), full history
EXTRA = dict(frame=[1.0, 2.5, 4.0, 6.0], lstm=[0.0, 0.5, 1.0, 2.0], cfc=[0.1, 0.6, 1.1, 2.1],
             ltc=[3.0, 3.5, 4.0, 5.0], transformer=[-0.0, 0.0, 0.2, 1.0])


def fake(tmp, arms, seeds=3, n=900):
    rng = np.random.default_rng(4)
    y = rng.normal(0, 15, (n, 4))
    cluster = np.repeat(np.arange(6), n // 6)
    session = np.array([f"2017030{c}1200" for c in cluster])
    for arm in arms:
        for s in range(seeds):
            preds = {}
            for keep, draws in ((1.0, 1), (0.5, 2), (0.25, 2)):
                for m in range(draws):
                    noise = rng.normal(0, 4, (n, 4))
                    extra = np.array(EXTRA[arm], dtype=float)
                    if arm != "frame":                    # dropping history costs more at longer horizons
                        extra = extra + (1 - keep) * np.array([0.2, 0.6, 1.0, 1.5])
                    preds[f"k{keep:g}_m{m}"] = y + noise + np.sign(noise) * extra
            np.savez(os.path.join(tmp, f"pred_{arm}_s{s}_val.npz"), y=y, cluster=cluster, session=session,
                     horizons=np.array(Z.HORIZONS), **preds)


def test_all_questions_on_a_known_structure(tmp_path):
    fake(str(tmp_path), Z.ARMS)
    res = Z.analyze(str(tmp_path), "val", n_boot=3000)
    assert res["missing"] == [] and res["anchors"] == 900 and res["clusters"] == 6
    # Q1: gain is 1.0 at 0 s and 3.0 at 1 s by construction
    assert res["q1"]["route_cluster"]["verdict"] == "yes"
    assert abs(res["q1"]["route_cluster"]["estimate"] - 2.0) < 0.3
    assert abs(res["q1"]["history_gain_by_horizon"]["0s"] - 1.0) < 0.3
    # the secondary family: 3 arms x 2 conditions, plus 4 sequence arms
    names = [t["test"] for t in res["secondary"]]
    assert res["secondary_m"] == len(names) == 10
    by = {t["test"]: t for t in res["secondary"]}
    assert by["Q2 ltc-lstm 1s keep_1"]["verdict_holm"] == "different"
    assert abs(by["Q2 ltc-lstm 1s keep_1"]["estimate"] - 3.0) < 0.3
    assert by["Q2 cfc-lstm 1s keep_1"]["verdict_holm"] != "different" or abs(by["Q2 cfc-lstm 1s keep_1"]["estimate"]) < 0.5
    assert by["Q2 transformer-lstm 1s keep_1"]["estimate"] < 0
    # Q3: extra degradation at 1 s over 0 s is (1.0 - 0.2) * 0.75 = 0.6 by construction
    assert abs(by["Q3 lstm extra degradation at 1s"]["estimate"] - 0.6) < 0.3
    assert all(t["p_equiv"] is None for t in res["secondary"] if t["test"].startswith("Q3"))
    assert all(t["p_diff_holm"] >= t["p_diff"] for t in res["secondary"])
    # Q4: the table has every arm, condition and horizon; the frame arm ignores the condition
    assert set(res["table"]) == set(Z.ARMS)
    assert len(res["table"]["lstm"]["keep_0.25"]["mean"]) == 4
    assert res["baselines"]["true_persistence"][0] == 0.0
    assert len(res["table"]["lstm"]["nowcast_as_forecast"]) == 4


def test_missing_arms_are_reported_and_can_be_required(tmp_path):
    fake(str(tmp_path), ["frame", "lstm", "cfc"])
    res = Z.analyze(str(tmp_path), "val", n_boot=3000)
    assert res["missing"] == ["ltc", "transformer"]
    assert res["secondary_m"] == 2 + 2                      # cfc at two conditions; Q3 for lstm and cfc
    with pytest.raises(SystemExit):
        Z.analyze(str(tmp_path), "val", n_boot=3000, require_all=True)


def test_the_reference_arm_is_required(tmp_path):
    fake(str(tmp_path), ["frame", "cfc"])
    with pytest.raises(SystemExit):
        Z.analyze(str(tmp_path), "val", n_boot=3000)
