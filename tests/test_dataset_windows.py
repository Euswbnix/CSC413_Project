"""Window legality, batch assembly and the rollout that produces every final number.

Uses a FABRICATED processed directory rather than real data, so these invariants run in the
default suite. The cases are chosen to be adversarial: a timestamp gap in the middle of a
split, and a missing image row inside the window range.
"""

import json

import numpy as np
import pytest
import torch

from data.dataset import (SteeringData, TRAIN_STRIDE_EQUIVALENT, center_crop,
                          rollout_chunks, train_batches, window_batches)

N, T = 400, 16
SEGMENTS = [[0, 150], [150, 400]]          # a recording gap at 150
SPLITS = {"train": [0, 200], "val": [220, 260], "test": [280, 400]}
MISSING = [95]                             # an image that exists in data.txt but not on disk


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("processed")
    # frames[i, 0, 0, 0] == i % 256, so a gather off by one frame is visible.
    frames = np.zeros((N, 66, 240, 3), dtype=np.uint8)
    # Channel 1 carries a horizontal gradient so a crop shift is VISIBLE -- without it the
    # frames are uniform, every shift produces the same tensor, and the test that checks
    # "different epoch_seed => different batch" passes vacuously.
    frames[:, :, :, 1] = (np.arange(240) % 256).astype(np.uint8)[None, None, :]
    # Channel 0, row 0, col 0 is the frame-index marker used by the gather-alignment test.
    frames[:, 0, 0, 0] = np.arange(N) % 256
    np.save(d / "frames.npy", frames)
    np.save(d / "angles_deg.npy", (np.arange(N) * 0.5 - 50).astype(np.float32))
    (d / "manifest.json").write_text(json.dumps({
        "n_frames": N, "stored_hw": [66, 240], "model_w": 200, "src_crop_rows": 150,
        "median_dt_s": 1 / 30, "fps_measured": 30.0, "has_timestamps": True,
        "gap_seconds": 0.5, "segments": SEGMENTS, "split_buffer": 20,
        "split_fractions": [0.7, 0.15, 0.15], "splits": SPLITS, "missing_rows": MISSING,
        "target_mean_deg": 0.0, "target_std_deg": 10.0,
        "pixel_mean": [0.5] * 3, "pixel_std": [0.25] * 3,
        "normalisation_scope": "train split only",
    }))
    return SteeringData(d, device="cpu", pin=False)


# ------------------------------------------------------------------ window legality ----

def test_no_window_crosses_a_split_boundary(data):
    for split, (lo, hi) in SPLITS.items():
        for s in data.window_starts(split, T).tolist():
            assert lo <= s and s + T <= hi, f"{split}: window [{s},{s+T}) escapes [{lo},{hi})"


def test_no_window_crosses_a_segment_boundary(data):
    """A 16-frame window straddling a multi-second recording gap is a legal-looking sample
    with a discontinuous label. The buffer guards split boundaries only; this guards gaps."""
    for split in SPLITS:
        for s in data.window_starts(split, T).tolist():
            inside = [(a, b) for a, b in SEGMENTS if a <= s and s + T <= b]
            assert inside, f"window [{s},{s+T}) spans a segment gap"


def test_no_window_contains_a_missing_row(data):
    for split in SPLITS:
        for s in data.window_starts(split, T).tolist():
            assert not set(range(s, s + T)) & set(MISSING), (
                f"window [{s},{s+T}) contains a row whose image is absent -- training on a "
                f"black frame with a real label looks like an architecture result")


def test_window_count_is_exactly_what_the_constraints_imply(data):
    # segment [0,150) within train: starts 0..134, minus the 16 starts whose window covers 95
    # segment [150,200) within train: starts 150..184
    assert len(data.window_starts("train", T)) == (135 - 16) + 35 == 154


def test_windows_per_epoch_is_the_stride_equivalent(data):
    n = len(data.window_starts("train", T))
    assert data.windows_per_epoch("train", T) == n // TRAIN_STRIDE_EQUIVALENT


# ------------------------------------------------------------------- batch assembly ----

def test_gather_is_frame_aligned(data):
    """frames[i,0,0,0] == i % 256, so this fails on an off-by-one in the index arithmetic."""
    starts = torch.tensor([0, 37, 150])
    f, y = data.gather(starts, T)
    for b, s in enumerate(starts.tolist()):
        got = (f[b, :, 0, 0, 0] * 255).round().long()
        assert got.tolist() == [(s + t) % 256 for t in range(T)]
        torch.testing.assert_close(y[b], data.angles_deg[s:s + T], rtol=0, atol=0)


def test_standardise_round_trips(data):
    y = torch.randn(4, T) * 30
    torch.testing.assert_close(data.to_degrees(data.standardise(y)), y, rtol=0, atol=1e-5)


def test_validation_batches_are_deterministic_and_centre_cropped(data):
    a = [tuple(x.shape) for x, _ in window_batches(data, "val", T, 8)]
    b = [tuple(x.shape) for x, _ in window_batches(data, "val", T, 8)]
    assert a == b and all(s[-1] == data.model_w for s in a)


def test_training_batches_are_reproducible_given_the_epoch_seed(data):
    def run(seed):
        return [(f.clone(), y.clone()) for f, y in
                train_batches(data, T, 4, k_deg_per_px=0.1, epoch_seed=seed)][:3]
    for (f1, y1), (f2, y2) in zip(run(7), run(7)):
        torch.testing.assert_close(f1, f2, rtol=0, atol=0)
        torch.testing.assert_close(y1, y2, rtol=0, atol=0)
    differs = any((a[0] - b[0]).abs().max() > 0 for a, b in zip(run(7), run(8)))
    assert differs, "a different epoch_seed produced identical batches"


def test_training_batches_are_augmented_and_validation_batches_are_not(data):
    tf = next(iter(train_batches(data, T, 4, k_deg_per_px=0.0, epoch_seed=1)))[0]
    vf = next(iter(window_batches(data, "train", T, 4)))[0]
    assert tf.shape[-1] == vf.shape[-1] == data.model_w
    # validation is the exact centre crop; training is not (some window is shifted)
    raw, _ = data.gather(data.window_starts("train", T)[:4], T)
    torch.testing.assert_close(vf, center_crop(raw, data.model_w), rtol=0, atol=0)


# ------------------------------------------------------------------------- rollout ----

@pytest.mark.parametrize("split", list(SPLITS))
@pytest.mark.parametrize("chunk", [16, 64, 256])
def test_chunked_rollout_covers_every_frame_once(data, split, chunk):
    """The final-number path. Chunking with `hx` carried across is equivalent to one long
    forward pass, and unlike strided windows it counts each frame exactly once -- no double
    counting, none discarded."""
    lo, hi = SPLITS[split]
    expected = set()
    for a, b in SEGMENTS:
        expected |= set(range(max(a, lo), min(b, hi)))
    seen = []
    for (s, e), f, y in rollout_chunks(data, split, chunk=chunk):
        assert f.shape[1] == e - s and f.shape[-1] == data.model_w
        seen.extend(range(s, e))
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen)), "a frame was evaluated more than once"


def test_rollout_never_spans_a_segment_gap(data):
    for (s, e), _, _ in rollout_chunks(data, "train", chunk=256):
        assert any(a <= s and e <= b for a, b in SEGMENTS), f"chunk [{s},{e}) spans a gap"
