"""train.py -> evaluate.py end to end on CPU, plus the rollout property that matters.

Small and synthetic on purpose: this gates the WIRING (masked loss, selection metric, run
directory, checkpoint round-trip, rollout coverage), none of which needs real data or a GPU.
Numerical quality is established by real runs, not here.
"""

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow

N, T = 1200, 8
SEGMENTS = [[0, 600], [600, N]]          # a gap at 600 -- state must not cross it
SPLITS = {"train": [0, 700], "val": [720, 940], "test": [960, N]}


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    d = tmp_path_factory.mktemp("proc")
    rng = np.random.default_rng(0)
    frames = (rng.random((N, 66, 240, 3)) * 255).astype(np.uint8)
    ang = (np.cumsum(rng.normal(0, 2, N)) % 80 - 40).astype(np.float32)
    drop = np.zeros(N, bool); drop[[100, 101, 900]] = True
    np.save(d / "frames.npy", frames)
    np.save(d / "angles_deg.npy", ang)
    np.save(d / "label_dropout.npy", drop)
    np.save(d / "dt_norm.npy", rng.uniform(0.5, 1.8, N).astype(np.float32))
    tr = ang[SPLITS["train"][0]:SPLITS["train"][1]][~drop[:SPLITS["train"][1]]]
    (d / "manifest.json").write_text(json.dumps({
        "n_frames": N, "stored_hw": [66, 240], "model_w": 200, "src_crop_rows": 150,
        "median_dt_s": 0.057, "fps_measured": 17.5, "has_timestamps": True,
        "gap_seconds": 0.5, "segments": SEGMENTS, "split_buffer": 20,
        "split_fractions": [0.6, 0.2, 0.2], "splits": SPLITS,
        "missing_rows": [], "n_label_dropouts": int(drop.sum()),
        "target_mean_deg": float(tr.mean()), "target_std_deg": float(tr.std()),
        "pixel_mean": [0.5] * 3, "pixel_std": [0.25] * 3,
        "normalisation_scope": "train split only",
    }))
    return d


def run(cmd):
    r = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@pytest.fixture(scope="module")
def trained(processed, tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    out = run(["train.py", "--arm", "cfc", "--seed", "0", "--T", str(T), "--epochs", "2",
               "--max-steps", "3", "--device", "cpu", "--processed", str(processed),
               "--runs", str(runs), "--name", "t"])
    return runs / "t", out


def test_run_directory_is_self_describing(trained):
    d, _ = trained
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["arm"] == "cfc" and cfg["seed"] == 0
    assert cfg["git_sha"] and cfg["versions"]["torch"]
    assert cfg["arm_params"]["total"] == 193_141
    # importlib.metadata, not module.__version__ -- the ncps wheel declares a version that
    # does not exist on PyPI, and a config.json citing it would be unreproducible
    assert cfg["versions"]["ncps"] != "0.0.2"
    assert (d / "metrics.csv").exists() and (d / "checkpoints" / "best.pt").exists()


def test_selection_is_on_macro_skill_not_global_mse(trained):
    """MUST-FIRE: both are logged so the counterfactual is free, and the checkpoint must
    follow the skill column. Selecting on MSE picks the most shrunken checkpoint of every
    run, undoing the reason the project bins at all."""
    d, _ = trained
    rows = list(csv_rows(d / "metrics.csv"))
    assert {"val_macro_skill", "val_mse_std"} <= set(rows[0])
    best = max(rows, key=lambda r: float(r["val_macro_skill"]))
    saved = torch.load(d / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    assert saved["epoch"] == int(best["epoch"])


def csv_rows(p):
    import csv as _csv
    with p.open() as fh:
        yield from _csv.DictReader(fh)


def test_evaluate_covers_the_split_once_and_writes_its_numbers(trained, processed):
    d, _ = trained
    out = run(["evaluate.py", str(d), "--split", "test", "--device", "cpu",
               "--processed", str(processed), "--chunk", "64"])
    m = json.loads((d / "final_metrics_test.json").read_text())
    lo, hi = SPLITS["test"]
    assert m["n_frames"] == hi - lo, "the rollout must cover the split exactly once"
    assert len(m["mae_by_window_position"]) == T
    assert set(m["rollout"]["bins"]) >= {"straight [0,5)", "curve [15,inf)"}
    assert "macro skill" in out


def test_rollout_resets_the_state_at_every_segment_boundary(processed, monkeypatch):
    """Tests the RESET MECHANISM, not a downstream proxy for it.

    The first version of this test compared predictions with and without the break and
    asserted they differ. They differed by 1.2e-7 -- an untrained recurrent state has decayed
    to something input-driven and carries almost no history, so the proxy was washed out and
    the test would have passed over a real bug. It did pass over one: the reset fired on
    "did the frame index jump?", and adjacent segments share a boundary, so it never fired.
    """
    import evaluate as ev
    from data.dataset import SteeringData
    from models.interface import build_arm

    seen = []
    model = build_arm("cfc").eval()
    real = model.forward

    def spy(frames, dt=None, hx=None):
        seen.append(hx is None)
        return real(frames, dt=dt, hx=hx)

    monkeypatch.setattr(model, "forward", spy)
    d = SteeringData(processed, device="cpu", pin=False)
    ev.rollout_predictions(model, d, "train", 64, False)

    chunks = [(seg, s, e) for (seg, s, e), _ in
              __import__("data.dataset", fromlist=["x"]).rollout_chunks(d, "train", chunk=64)]
    assert len(seen) == len(chunks)
    firsts = {seg: i for i, (seg, _, _) in reversed(list(enumerate(chunks)))}
    for i, (seg, s, e) in enumerate(chunks):
        expect_reset = (i == firsts[seg])
        assert seen[i] == expect_reset, (
            f"chunk {i} (segment {seg}, frames {s}-{e}): hx was "
            f"{'None' if seen[i] else 'carried'}, expected "
            f"{'None' if expect_reset else 'carried'}")
    assert sum(seen) == len(set(seg for seg, _, _ in chunks)) == 2


def test_reporting_tools_close_the_loop(trained, processed):
    """train -> evaluate -> tables -> figures, so the command that regenerates the README's
    evidence is exercised rather than assumed. The protocol's rule is that experiments do not
    start until this path works; a test is the only way that rule means anything."""
    d, _ = trained
    run(["evaluate.py", str(d), "--split", "test", "--device", "cpu",
         "--processed", str(processed), "--chunk", "64"])

    md = run(["scripts/make_tables.py", "--runs", str(d.parent), "--split", "test"])
    assert "predict-0" in md and "persistence" in md
    assert "events" in md, "every bin column must carry its independent turn-event count"
    assert "(1.00x)" in md, "predict-0 must appear as its own reference ratio"
    assert "cfc" in md

    figs = d.parent / "figs"
    out = run(["scripts/make_figures.py", "--runs", str(d.parent), "--split", "test",
               "--out", str(figs)])
    made = sorted(p.name for p in figs.glob("*.png"))
    assert made == ["bin_mae.png", "mae_by_window_position.png", "training_curves.png"], made
    assert all(p.stat().st_size > 5000 for p in figs.glob("*.png")), "a figure came out empty"


def test_make_tables_refuses_rather_than_emitting_an_empty_table(tmp_path):
    """MUST-FIRE: an empty runs/ must be an error, not a table with no rows. A silently
    empty table pasted into the README reads as a result."""
    import subprocess as sp
    empty = tmp_path / "none"; empty.mkdir()
    r = sp.run([sys.executable, "scripts/make_tables.py", "--runs", str(empty)],
               cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "run evaluate.py first" in r.stderr
