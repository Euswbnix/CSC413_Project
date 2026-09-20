"""Window legality, batch assembly and the rollout that produces every final number.

Uses a FABRICATED processed directory rather than real data, so these invariants run in the
default suite. The cases are chosen to be adversarial: a timestamp gap in the middle of a
split, and a missing image row inside the window range.
"""

import json

import numpy as np
import pytest
import torch

from data.dataset import (Batch, SteeringData, TRAIN_STRIDE_EQUIVALENT, center_crop,
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
    f, y, v, _ = data.gather(starts, T)
    for b, s in enumerate(starts.tolist()):
        got = (f[b, :, 0, 0, 0] * 255).round().long()
        assert got.tolist() == [(s + t) % 256 for t in range(T)]
        torch.testing.assert_close(y[b], data.angles_deg[s:s + T], rtol=0, atol=0)


def test_standardise_round_trips(data):
    y = torch.randn(4, T) * 30
    torch.testing.assert_close(data.to_degrees(data.standardise(y)), y, rtol=0, atol=1e-5)


def test_validation_batches_are_deterministic_and_centre_cropped(data):
    a = [tuple(b.frames.shape) for b in window_batches(data, "val", T, 8)]
    b = [tuple(x.frames.shape) for x in window_batches(data, "val", T, 8)]
    assert a == b and all(s[-1] == data.model_w for s in a)


def test_training_batches_are_reproducible_given_the_epoch_seed(data):
    def run(seed):
        return [(b.frames.clone(), b.y.clone()) for b in
                train_batches(data, T, 4, k_deg_per_px=0.1, epoch_seed=seed)][:3]
    for (f1, y1), (f2, y2) in zip(run(7), run(7)):
        torch.testing.assert_close(f1, f2, rtol=0, atol=0)
        torch.testing.assert_close(y1, y2, rtol=0, atol=0)
    differs = any((a[0] - b[0]).abs().max() > 0 for a, b in zip(run(7), run(8)))
    assert differs, "a different epoch_seed produced identical batches"


def test_training_batches_are_augmented_and_validation_batches_are_not(data):
    tf = next(iter(train_batches(data, T, 4, k_deg_per_px=0.0, epoch_seed=1))).frames
    vf = next(iter(window_batches(data, "train", T, 4))).frames
    assert tf.shape[-1] == vf.shape[-1] == data.model_w
    # validation is the exact centre crop; training is not (some window is shifted)
    raw = data.gather(data.window_starts("train", T)[:4], T).frames
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
    for (_, s, e), b in rollout_chunks(data, split, chunk=chunk):
        f = b.frames
        assert f.shape[1] == e - s and f.shape[-1] == data.model_w
        seen.extend(range(s, e))
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen)), "a frame was evaluated more than once"


def test_rollout_never_spans_a_segment_gap(data):
    for (_, s, e), _ in rollout_chunks(data, "train", chunk=256):
        assert any(a <= s and e <= b for a, b in SEGMENTS), f"chunk [{s},{e}) spans a gap"


def test_validity_and_dt_are_fields_of_every_batch(data):
    """Neither is optional -- masking the loss is required and dt is the subject of the
    project's second question -- so both are fields a caller must handle, not attributes to
    remember to look up."""
    b = data.gather(torch.tensor([0, 50]), T)
    assert isinstance(b, Batch) and len(b) == 4
    assert b.valid.shape == (2, T) and b.valid.dtype == torch.bool and b.valid.all()
    assert b.dt.shape == (2, T)
    for bb in (next(iter(train_batches(data, T, 4, k_deg_per_px=0.1, epoch_seed=0))),
               next(iter(window_batches(data, "val", T, 4)))):
        assert isinstance(bb, Batch) and bb.valid.dtype == torch.bool
    (_, rb) = next(iter(rollout_chunks(data, "test", chunk=64)))
    assert isinstance(rb, Batch)


def test_dt_defaults_to_ones_when_the_file_is_absent(data):
    """MUST-FIRE: a silently-missing dt file must give a constant, not a crash and not
    garbage -- and the constant must be 1.0, because the gradient balance between the time
    head and the offset head is exactly |dt|."""
    assert torch.allclose(data.gather(torch.tensor([0]), T).dt, torch.ones(1, T))


def test_rollout_identifies_the_segment_rather_than_leaving_it_to_be_inferred(data):
    """MUST-FIRE: adjacent segments share a boundary -- [0,150) then [150,400) -- so the
    chunk after the break starts exactly where the previous one ended. A caller testing
    "did the index jump?" would see no jump and carry the hidden state across a recording
    gap the camera never saw."""
    got = [(seg, s, e) for (seg, s, e), _ in rollout_chunks(data, "train", chunk=256)]
    assert [g[0] for g in got] == [0, 1], "both segments must be visited and distinguishable"
    (seg_a, _, end_a), (seg_b, start_b, _) = got[0], got[1]
    assert end_a == start_b, "the fixture's segments are adjacent; that is the trap"
    assert seg_a != seg_b, "and the segment index is the only thing that reveals it"
