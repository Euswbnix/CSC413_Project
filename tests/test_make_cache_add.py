"""hdd/make_cache.py --add: the test split is appended once, before the single test evaluation,
without touching a byte of the cached train/val sessions -- on synthetic features."""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "hdd", "make_cache.py")


def _session(path, n=90, seed=0):
    rng = np.random.default_rng(seed)
    np.savez(path, feats=rng.normal(size=(n, 4)).astype(np.float16), kept=np.arange(n, dtype=np.int32),
             t_cam=np.arange(n) / 30.0, steer=rng.normal(size=n).astype(np.float32),
             speed_mps=np.full(n, 5.0, dtype=np.float32))


def _setup(tmp, with_test_features=True):
    feats, split, cache = (os.path.join(tmp, d) for d in ("dinov2_s1", "split_a", "cache"))
    os.makedirs(feats)
    os.makedirs(split)
    sessions = {"201701010000": ("train", 0), "201701020000": ("val", 1),
                "201701030000": ("test", 2), "201701040000": ("test", 3)}
    for i, (s, (sp, _)) in enumerate(sessions.items()):
        if sp != "test" or with_test_features:
            _session(os.path.join(feats, f"{s}.npz"), seed=i)
    json.dump({"sessions": {s: {"split": sp, "cluster": c} for s, (sp, c) in sessions.items()}},
              open(os.path.join(split, "split.json"), "w"))
    return feats, split, cache


def _run(*args):
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)


def _digests(cache):
    return {f: hashlib.sha256(open(os.path.join(cache, f), "rb").read()).hexdigest()
            for f in sorted(os.listdir(cache)) if f.endswith(".npy")}


def test_add_appends_test_and_leaves_train_val_untouched(tmp_path):
    feats, split, cache = _setup(str(tmp_path))
    assert _run("--features", feats, "--split", split, "--out", cache).returncode == 0
    before = _digests(cache)
    mtimes = {f: os.stat(os.path.join(cache, f)).st_mtime_ns for f in before}
    r = _run("--features", feats, "--split", split, "--out", cache, "--splits", "test", "--add")
    assert r.returncode == 0, r.stderr
    after = _digests(cache)
    assert all(after[f] == h for f, h in before.items())
    assert all(os.stat(os.path.join(cache, f)).st_mtime_ns == m for f, m in mtimes.items())
    idx = json.load(open(os.path.join(cache, "index.json")))["sessions"]
    assert sorted(m["split"] for m in idx.values()) == ["test", "test", "train", "val"]
    assert not os.path.exists(os.path.join(cache, "index.json.tmp"))


def test_add_refuses_a_missing_test_session(tmp_path):
    feats, split, cache = _setup(str(tmp_path), with_test_features=False)
    assert _run("--features", feats, "--split", split, "--out", cache).returncode == 0
    r = _run("--features", feats, "--split", split, "--out", cache, "--splits", "test", "--add")
    assert r.returncode != 0 and "no features for test session" in r.stderr
    idx = json.load(open(os.path.join(cache, "index.json")))["sessions"]
    assert all(m["split"] != "test" for m in idx.values())


def test_add_refuses_other_features(tmp_path):
    feats, split, cache = _setup(str(tmp_path))
    assert _run("--features", feats, "--split", split, "--out", cache).returncode == 0
    other = os.path.join(str(tmp_path), "resnet_s1")
    os.rename(feats, other)
    r = _run("--features", other, "--split", split, "--out", cache, "--splits", "test", "--add")
    assert r.returncode != 0 and "refusing to mix" in r.stderr
