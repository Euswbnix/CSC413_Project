"""REVIEW B3's gate. The most important invariant in the data pipeline, and the one whose
violation is silent, asymmetric, and produces a publishable-looking false conclusion.

One transform draw per WINDOW, applied identically to all T frames. The test suite asserts
both halves: that the real augmenter holds the invariant, and that `per_frame_bug=True`
breaks it -- so the bug-control runs are a measured instrument rather than a hope.
"""

import torch

from data.dataset import augment_batch, center_crop

B, T, C, H, W, MW = 6, 16, 3, 66, 240, 200
OFF = dict(brightness=0.0, shadow_prob=0.0, flip_prob=0.0)


def constant_batch(value=0.5, label=12.0):
    """Every frame in every window identical, every label identical. Any variation across T
    in the output therefore comes from the augmenter, not from the input."""
    return (torch.full((B, T, C, H, W), value), torch.full((B, T), label))


def g(seed=0):
    return torch.Generator().manual_seed(seed)


# ----------------------------------------------------- the invariant, both halves ----

def test_one_transform_per_window_frames_identical_across_T():
    f, y = constant_batch()
    out, _ = augment_batch(f, y, k_deg_per_px=0.1, model_w=MW, generator=g())
    for t in range(1, T):
        torch.testing.assert_close(out[:, t], out[:, 0], rtol=0, atol=0)


def test_one_transform_per_window_label_corrections_identical_across_T():
    f, y = constant_batch()
    _, lab = augment_batch(f, y, k_deg_per_px=0.1, model_w=MW, generator=g())
    for t in range(1, T):
        torch.testing.assert_close(lab[:, t], lab[:, 0], rtol=0, atol=0)


def test_the_per_frame_bug_breaks_both_halves():
    """If this ever starts passing, the deliberate control has stopped being a control and
    MODEL.md / the README paragraph about it are stale."""
    f, y = constant_batch()
    out, lab = augment_batch(f, y, k_deg_per_px=0.1, model_w=MW, generator=g(),
                             per_frame_bug=True)
    frames_vary = any((out[:, t] - out[:, 0]).abs().max() > 0 for t in range(1, T))
    labels_vary = any((lab[:, t] - lab[:, 0]).abs().max() > 0 for t in range(1, T))
    assert frames_vary, "per-frame bug did not vary frames across T"
    assert labels_vary, "per-frame bug did not vary label corrections across T"


# -------------------------------------------------------------------------- flip ----

def test_flip_negates_the_whole_label_sequence_exactly():
    f = torch.rand(B, T, C, H, W, generator=g(1))
    y = torch.randn(B, T, generator=g(2)) * 20
    plain, lab_plain = augment_batch(f, y, k_deg_per_px=0.0, model_w=MW, generator=g(3),
                                     **{**OFF, "flip_prob": 0.0})
    flipped, lab_flip = augment_batch(f, y, k_deg_per_px=0.0, model_w=MW, generator=g(3),
                                      **{**OFF, "flip_prob": 1.0})
    torch.testing.assert_close(lab_flip, -lab_plain, rtol=0, atol=0)
    torch.testing.assert_close(flipped, plain.flip(-1), rtol=0, atol=0)


def test_flip_is_all_or_nothing_within_a_window():
    """A per-frame flip would negate the labels of only SOME frames, producing a physically
    impossible sequence that still trains."""
    f = torch.rand(B, T, C, H, W, generator=g(4))
    y = torch.full((B, T), 7.0)
    _, lab = augment_batch(f, y, k_deg_per_px=0.0, model_w=MW, generator=g(5), **OFF)
    signs = torch.sign(lab)
    assert (signs == signs[:, :1]).all(), "sign of the label varies within a window"


# ------------------------------------------------------------------------- shift ----

def test_shift_correction_is_linear_in_the_displacement_and_constant_across_T():
    k = 0.2
    f, y = constant_batch(label=0.0)
    for seed in range(8):
        out, lab = augment_batch(f, y, k_deg_per_px=k, model_w=MW, generator=g(seed), **OFF)
        d = -lab[:, 0] / k                       # recovered displacement, px
        assert torch.allclose(d, d.round(), atol=1e-4), "displacement is not an integer"
        assert d.abs().max() <= (W - MW) // 2, "displacement exceeds the stored margin"
        torch.testing.assert_close(lab, lab[:, :1].expand(B, T), rtol=0, atol=0)


def test_zero_coefficient_leaves_labels_untouched():
    """k = 0 is a point in the hyperparameter sweep, and it must be an exact no-op on the
    labels or the sweep measures two things at once."""
    f = torch.rand(B, T, C, H, W, generator=g(6))
    y = torch.randn(B, T, generator=g(7)) * 20
    _, lab = augment_batch(f, y, k_deg_per_px=0.0, model_w=MW, generator=g(8), **OFF)
    torch.testing.assert_close(lab, y, rtol=0, atol=0)


def test_output_width_is_the_model_width_and_crop_stays_in_range():
    f = torch.rand(2, 4, C, H, W, generator=g(9))
    out, _ = augment_batch(f, torch.zeros(2, 4), k_deg_per_px=0.1, model_w=MW, generator=g(10))
    assert out.shape == (2, 4, C, H, MW)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_center_crop_is_the_middle_columns():
    f = torch.arange(W, dtype=torch.float32).view(1, 1, 1, 1, W).expand(1, 1, C, H, W)
    torch.testing.assert_close(center_crop(f, MW)[0, 0, 0, 0],
                               torch.arange(20, 220, dtype=torch.float32), rtol=0, atol=0)


def test_photometric_augmentation_does_not_touch_the_label():
    """Brightness and shadow are photometric: they belong in the SEEN-IN-TRAINING column of
    the OOD table and must not move the target."""
    f = torch.rand(B, T, C, H, W, generator=g(11))
    y = torch.randn(B, T, generator=g(12)) * 20
    _, lab = augment_batch(f, y, k_deg_per_px=0.0, model_w=MW, generator=g(13),
                           brightness=0.4, shadow_prob=1.0, flip_prob=0.0)
    torch.testing.assert_close(lab, y, rtol=0, atol=0)


def test_aug_none_really_means_none():
    """MUST-FIRE: `--aug none` is one arm of the ablation. If the switch were cosmetic the
    row would compare full augmentation against most of it and the Advanced Concept claim
    would rest on a difference that was never controlled."""
    import torch as _t
    from data.dataset import augment_batch as _ab
    f = _t.rand(4, 8, C, H, W, generator=g(20))
    y = _t.randn(4, 8, generator=g(21)) * 20
    out, lab = _ab(f, y, k_deg_per_px=0.0, model_w=MW, generator=g(22),
                   brightness=0.0, shadow_prob=0.0, flip_prob=0.0)
    torch.testing.assert_close(lab, y, rtol=0, atol=0)
    torch.testing.assert_close(out, center_crop(f, MW), rtol=0, atol=0)
    on, _ = _ab(f, y, k_deg_per_px=0.0, model_w=MW, generator=g(22))
    assert (on - center_crop(f, MW)).abs().max() > 0, "augmentation ON changed nothing"
