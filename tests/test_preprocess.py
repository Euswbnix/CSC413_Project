"""End-to-end: synthetic JPEGs -> uint8 memmap -> manifest. Marked slow (it decodes 5,000
images), but it is the only test that exercises the real decode path and the alignment rule.

The synthetic frames encode their own row index as the brightness of the bottom 150 rows --
the region preprocess keeps -- so a uniform frame/label off-by-one is DETECTABLE here. That
matters because every statistic in the EDA passes such a misalignment, and a scrambled time
axis produces a false "the recurrent layer adds nothing" that survives the overfit gate.
"""

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.slow

ROOT = pathlib.Path(__file__).resolve().parents[1]
N = 5000                       # > 4000, or the 300-frame buffer leaves an empty val split
DROPPED_ROW = 1234             # a data.txt line whose image we delete on purpose


def marker(i):
    return (i * 7) % 200 + 28  # stays clear of 0 and 255 so JPEG never clips it


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    from PIL import Image
    raw = tmp_path_factory.mktemp("raw")
    out = tmp_path_factory.mktemp("processed")
    rng = np.random.default_rng(0)
    ang = np.cumsum(rng.normal(0, 1.2, N)) % 120 - 60
    lines = []
    for i in range(N):
        img = np.empty((256, 455, 3), np.uint8)
        img[:106] = rng.integers(0, 255, (106, 455, 3), dtype=np.uint8)   # sky: discarded
        img[106:] = marker(i)                                            # kept by the crop
        Image.fromarray(img).save(raw / f"{i}.jpg", quality=92)
        ms = int((i * 1000 / 30) % 1000)
        sec = int(i / 30)
        lines.append(f"{i}.jpg {ang[i]:.6f},2018-03-12 "
                     f"{14 + sec // 3600:02d}:{(sec // 60) % 60:02d}:{sec % 60:02d}:{ms:03d}")
    (raw / "data.txt").write_text("\n".join(lines) + "\n")
    (raw / f"{DROPPED_ROW}.jpg").unlink()        # label line with no image

    r = subprocess.run([sys.executable, "data/preprocess.py", "--data-root", str(raw),
                        "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out, ang, r.stdout


def test_line_i_is_row_i(processed):
    """The alignment rule, tested rather than asserted in a comment. Also checks the
    NEIGHBOURS: a uniform off-by-one would still give a plausible mean, so the test asks
    which candidate index the observed brightness matches BEST."""
    out, _, _ = processed
    frames = np.load(out / "frames.npy", mmap_mode="r")
    for i in (0, 1, 17, 999, 2500, N - 1):
        if i == DROPPED_ROW:
            continue
        observed = float(frames[i].mean())
        candidates = {j: abs(observed - marker(j)) for j in range(max(0, i - 4), min(N, i + 5))}
        best = min(candidates, key=candidates.get)
        assert best == i, (f"row {i} looks like source frame {best} "
                           f"(observed {observed:.1f}, expected {marker(i)})")
        assert candidates[i] < 4.0


def test_stored_shape_is_the_storage_width_not_the_model_width(processed):
    out, _, _ = processed
    frames = np.load(out / "frames.npy", mmap_mode="r")
    assert frames.shape == (N, 66, 240, 3) and frames.dtype == np.uint8
    m = json.loads((out / "manifest.json").read_text())
    assert m["stored_hw"] == [66, 240] and m["model_w"] == 200, (
        "the 40 px of extra width is what the shift augmentation shifts into; storing at the "
        "model width forces edge padding whose band width is a linear function of the label "
        "correction, which the model can regress instead of the road")


def test_missing_image_is_recorded_not_silently_zeroed(processed):
    out, _, stdout = processed
    m = json.loads((out / "manifest.json").read_text())
    assert m["missing_rows"] == [DROPPED_ROW]
    assert "had no image file" in stdout
    frames = np.load(out / "frames.npy", mmap_mode="r")
    assert frames[DROPPED_ROW].max() == 0


def test_labels_round_trip_and_normalisation_is_train_only(processed):
    out, ang, _ = processed
    saved = np.load(out / "angles_deg.npy")
    np.testing.assert_allclose(saved, ang.astype(np.float32), rtol=0, atol=1e-4)
    m = json.loads((out / "manifest.json").read_text())
    lo, hi = m["splits"]["train"]
    assert m["normalisation_scope"] == "train split only"
    assert abs(m["target_mean_deg"] - float(ang[lo:hi].mean())) < 1e-3
    assert abs(m["target_std_deg"] - float(ang[lo:hi].std())) < 1e-3
    # and it must NOT be the statistic over all frames -- that would leak test data
    assert abs(m["target_mean_deg"] - float(ang.mean())) > 1e-6


def test_splits_are_chronological_non_overlapping_and_buffered(processed):
    out, _, _ = processed
    m = json.loads((out / "manifest.json").read_text())
    (a0, a1), (b0, b1), (c0, c1) = (m["splits"][k] for k in ("train", "val", "test"))
    assert a0 < a1 <= b0 < b1 <= c0 < c1 == N
    assert b0 - a1 == m["split_buffer"] * 2 or b0 - a1 >= m["split_buffer"]
    assert c0 - b1 >= m["split_buffer"]
    assert m["split_buffer"] >= 300


def test_measured_frame_rate_is_recorded_and_not_assumed(processed):
    out, _, _ = processed
    m = json.loads((out / "manifest.json").read_text())
    assert m["has_timestamps"] and m["fps_measured"] is not None
    assert 29.0 < m["fps_measured"] < 31.0


def test_the_processed_directory_drives_the_dataset(processed):
    """The handoff: whatever preprocess writes must be directly loadable, or the two halves
    of the pipeline drift."""
    from data.dataset import SteeringData, rollout_chunks, train_batches
    out, _, _ = processed
    d = SteeringData(out, device="cpu", pin=False)
    starts = d.window_starts("train", 16)
    assert len(starts) > 0
    assert not any(DROPPED_ROW in range(s, s + 16) for s in starts.tolist())
    f, y = next(iter(train_batches(d, 16, 4, k_deg_per_px=0.1, epoch_seed=0)))
    assert f.shape == (4, 16, 3, 66, 200) and y.shape == (4, 16)
    covered = sum(e - s for (s, e), _, _ in rollout_chunks(d, "test", chunk=256))
    lo, hi = d.manifest["splits"]["test"]
    assert covered == hi - lo
