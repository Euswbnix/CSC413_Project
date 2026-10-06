"""Steering anticipation (docs/plan_2026-10-05_steering_anticipation.md): the future labels, the
anchors shared by every horizon, the no-history arm, and a tiny end-to-end run, all on synthetic
data."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hdd"))
import anticip_run as A  # noqa: E402
import d4_run as D  # noqa: E402


def steer_of(t):
    return 20.0 * np.sin(0.7 * t) + 3.0 * t


def test_labels_are_exact_on_a_regular_linear_series():
    t = 0.1 * np.arange(100)
    steer = 3.0 * t - 5.0
    anchors = np.array([0, 10, 50])
    Y = A.future_labels(t, steer, anchors)
    for k, h in enumerate(A.HORIZONS):
        assert np.allclose(Y[:, k], 3.0 * (t[anchors] + h) - 5.0, atol=1e-9)


def test_labels_interpolate_between_irregular_frames():
    t = np.array([0.0, 0.1, 0.2, 0.31, 0.38, 0.52, 0.6, 0.7])
    steer = np.array([0.0, 1.0, 2.0, 4.0, 5.0, 9.0, 10.0, 11.0])
    Y = A.future_labels(t, steer, np.array([0]), horizons=(0.0, 0.45))
    assert Y[0, 0] == 0.0
    assert np.isclose(Y[0, 1], 5.0 + (0.45 - 0.38) / (0.52 - 0.38) * 4.0)       # between 0.38 and 0.52


def test_labels_are_void_across_gaps_bad_values_backward_clocks_and_the_end():
    t = 0.1 * np.arange(60)
    steer = steer_of(t)
    gap = t.copy()
    gap[30:] += 0.5                                   # a 0.6 s hole after frame 29
    Y = A.future_labels(gap, steer, np.array([20, 26, 40]))
    assert np.isfinite(Y[0, 1]) and np.isnan(Y[0, 2])            # 20 -> +0.5 fine, +1.0 lands in the hole
    assert np.isnan(Y[1, 1])                                     # 26 -> +0.5 lands in the hole
    assert np.isfinite(Y[2, 1])                                  # after the hole everything is regular
    bad = steer.copy()
    bad[25] = np.nan
    assert np.isnan(A.future_labels(t, bad, np.array([20]))[0, 1])     # +0.5 s needs frame 25
    back = t.copy()
    back[35:] -= 2.0                                  # the clock steps backwards at frame 35
    Yb = A.future_labels(back, steer, np.array([30]))
    assert np.isnan(Yb[0, 2]) and np.isnan(Yb[0, 3])
    assert np.isnan(A.future_labels(t, steer, np.array([50]))[0, 3])   # +2 s is past the last frame
    assert np.isfinite(A.future_labels(t, steer, np.array([39]))[0, 3])


def make_cache(tmp, dim=16, n=160):
    cache = os.path.join(tmp, "cache")
    os.makedirs(cache)
    rng = np.random.default_rng(0)
    sessions = {"201701010000": ("train", 0), "201701020000": ("train", 1), "201701030000": ("val", 2)}
    for i, s in enumerate(sessions):
        t = 100.0 * i + 0.1 * np.arange(n) + rng.normal(0, 0.002, n).cumsum() * 0
        np.save(os.path.join(cache, f"{s}.feats.npy"), rng.normal(size=(n, dim)).astype(np.float16))
        np.save(os.path.join(cache, f"{s}.meta.npy"), np.column_stack([t, steer_of(t), np.full(n, 5.0)]))
    json.dump(dict(base_hz=10.0, stride=3, features="fake",
                   sessions={s: dict(split=sp, cluster=c, n=n) for s, (sp, c) in sessions.items()}),
              open(os.path.join(cache, "index.json"), "w"))
    return cache


def test_every_horizon_shares_the_same_anchors_and_batches_stay_aligned(tmp_path):
    cache = make_cache(str(tmp_path))
    data = A.AnticipCached(cache, "train", 30, "cpu")
    Y = data.labels()
    assert Y.shape == (data.n, 4) and np.isfinite(Y).all()
    assert len(data.clusters()) == len(data.sessions()) == data.n
    plain = D.Cached(cache, "train", 30, "cpu")
    assert data.n == plain.n - 2 * 20                 # the last 2 s of each session have no +2 s label
    for shuffle in (False, True):
        for f, t, y in data.batches(32, shuffle=shuffle, rng=np.random.default_rng(1)):
            now = t[:, -1].numpy()
            assert f.shape[1:] == (30, 16) and y.shape[1] == 4
            for k, h in enumerate(A.HORIZONS):
                assert np.allclose(y[:, k].numpy(), steer_of(now + h), atol=0.05), (shuffle, h)


def test_frame_arm_sees_only_the_current_frame():
    torch.manual_seed(0)
    m = D.Arm("frame", 16, n_out=4).double().eval()
    assert m.blocks()["temporal"] == 33_088
    f = torch.randn(3, 10, 16, dtype=torch.float64)
    valid = torch.arange(10)[None] < torch.tensor([10, 6, 3])[:, None]
    dt = torch.full((3, 10), 0.1, dtype=torch.float64)
    base = m(f, dt, valid)
    assert base.shape == (3, 4)
    f2 = f.clone()
    last = valid.long().sum(1) - 1
    keep = torch.zeros(3, 10, dtype=torch.bool)
    keep[torch.arange(3), last] = True
    f2[~keep] = torch.randn(int((~keep).sum()), 16, dtype=torch.float64)       # everything but the current frame
    assert torch.allclose(m(f2, dt, valid), base, atol=1e-12)


def run(*args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "hdd", "anticip_run.py"), *args],
                          capture_output=True, text=True)


@pytest.mark.parametrize("arm", ["frame", "lstm", "transformer"])
def test_a_tiny_run_trains_resumes_and_writes_four_horizon_predictions(tmp_path, arm):
    cache, out = make_cache(str(tmp_path)), os.path.join(str(tmp_path), "out")
    common = ["--cache", cache, "--arm", arm, "--out", out, "--batch", "64", "--mask-draws", "2", "--swanlab", "off"]
    r = run(*common, "--stage", "lr", "--lrs", "1e-3", "--tag", "_1e-3", "--epochs", "1")
    assert r.returncode == 0, r.stderr[-800:]
    grid = json.load(open(os.path.join(out, f"{arm}_lr_1e-3.json")))
    assert np.isfinite(grid["rows"][0]["mean_macro_mae"]) and grid["rows"][0]["epochs"] == 1
    r = run(*common, "--stage", "final", "--lr", "1e-3", "--seeds", "0", "--epochs", "2")   # resumes epoch 2
    assert r.returncode == 0, r.stderr[-800:]
    assert "resumed lr 0.001 seed 0 at epoch 1" in r.stdout
    z = np.load(os.path.join(out, f"pred_{arm}_s0_val.npz"))
    n = len(z["y"])
    assert z["y"].shape == (n, 4) and list(z["horizons"]) == list(A.HORIZONS)
    assert z["k1_m0"].shape == (n, 4) and z["k0.25_m1"].shape == (n, 4) and "k1_m1" not in z.files
    row = json.load(open(os.path.join(out, f"{arm}_s0_val.json")))
    assert len(row["conditions"]["keep_0.25"]) == 2 and len(row["conditions"]["keep_1"][0]["macro_mae"]) == 4
    assert row["blocks"]["readout"] == 260
    r = run(*common, "--stage", "final", "--lr", "1e-3", "--seeds", "0", "--epochs", "2")
    assert "already finished and evaluated" in r.stdout


def test_the_test_split_is_never_trained_on(tmp_path):
    cache, out = make_cache(str(tmp_path)), os.path.join(str(tmp_path), "out")
    idx = json.load(open(os.path.join(cache, "index.json")))
    idx["sessions"]["201701030000"]["split"] = "test"
    json.dump(idx, open(os.path.join(cache, "index.json"), "w"))
    r = run("--cache", cache, "--arm", "lstm", "--out", out, "--stage", "final", "--lr", "1e-3",
            "--seeds", "0", "--epochs", "1", "--split-name", "test", "--swanlab", "off")
    assert r.returncode != 0 and "refusing to train" in (r.stderr + r.stdout)
