"""Window sampling, batched GPU augmentation, and the stateful rollout used for evaluation.

No DataLoader and no workers. A step is: gather on a resident uint8 tensor ->
`.to(device, non_blocking=True)` -> `.float().div_(255)` -> augment on device. Removing
worker startup from 294 short runs is a side benefit; the decisive reason is CORRECTNESS,
below.

THE AUGMENTATION SHAPE IS THE POINT
-----------------------------------
Augmentation parameters are sampled at shape **(B,1,1,1,1)** against a (B,T,C,H,W) batch, so
"one transform per window, identical across all T frames" is what BROADCASTING DOES BY
DEFAULT and the buggy per-frame variant is the one that takes extra effort to write.

Written the usual way -- a torchvision transform inside `Dataset.__getitem__` -- each frame
draws its own shift, its own flip decision and its own brightness, so within one 16-frame
window the camera teleports laterally every ~34 ms and occasionally mirrors, and a flip
negates the labels of only some frames. That failure is silent (no error, loss still falls),
asymmetric (it corrupts every recurrent arm and leaves the single-frame CNN untouched), and
it manufactures exactly the false headline "temporal information does not help here".

`per_frame_bug=True` reproduces it deliberately, as a measuring instrument rather than an
accident: it is one argument, so the bug-control runs share every other code path with the
real runs. `tests/test_augmentation_windows.py` asserts the invariant and asserts that the
flag breaks it.

WINDOW SAMPLING
---------------
Uniformly random start indices, not a fixed stride grid. A fixed stride couples frame *i* to
only the within-window positions determined by `i mod stride`, which would contaminate the
per-position MAE curve that is the project's headline figure. The NUMBER of windows per epoch
is still the stride-4-equivalent count, because that number fixes the optimizer-step budget
and the budget is a confound in every comparison.

A window is legal only if it lies inside one split, inside one timestamp-gap segment, and
contains no row whose image was missing.
"""

import json
import pathlib
from typing import NamedTuple

import numpy as np
import torch

TRAIN_STRIDE_EQUIVALENT = 4          # windows/epoch = n_train_windows_at_stride_1 / 4


class Batch(NamedTuple):
    """Everything a step needs, in one object that cannot be partially unpacked by accident.

    `valid` and `dt` are fields rather than attributes to look up because neither is
    optional: masking the loss is required (973 frames carry a logging default of 0.0 in
    place of a real angle) and dt is the subject of the project's second question. A caller
    that forgets either has to write code that looks wrong.
    """
    frames: "torch.Tensor"      # (B,T,3,66,model_w) in [0,1]
    y: "torch.Tensor"           # (B,T) standardised unless stated
    valid: "torch.Tensor"       # (B,T) bool
    dt: "torch.Tensor"          # (B,T) normalised so the median step is 1.0


class SteeringData:
    def __init__(self, processed_dir, device="cpu", pin=True):
        d = pathlib.Path(processed_dir)
        self.manifest = json.loads((d / "manifest.json").read_text())
        self.angles_deg = torch.from_numpy(np.load(d / "angles_deg.npy")).float()
        # Explicit writable copy: np.load(mmap_mode="r") hands back a read-only view, and
        # torch.from_numpy on it warns and yields a tensor whose writes are undefined.
        frames = np.array(np.load(d / "frames.npy", mmap_mode="r"))
        # Resident in HOST ram and pinned, NOT in VRAM. The dataset fits in 32 GB nine times
        # over, but this workload is kernel-launch-bound rather than bandwidth-bound, so
        # concurrency is the only real speed lever and VRAM residency would halve the number
        # of jobs that fit on one card to optimise a transfer that was never the bottleneck.
        self.frames = torch.from_numpy(frames)   # ~2.99 GB resident in host RAM
        if pin and torch.cuda.is_available():
            self.frames = self.frames.pin_memory()
        self.device = torch.device(device)
        self.stored_h, self.stored_w = self.manifest["stored_hw"]
        self.model_w = self.manifest["model_w"]
        self.max_shift = (self.stored_w - self.model_w) // 2
        self.missing = set(self.manifest["missing_rows"])
        self.target_mean = self.manifest["target_mean_deg"]
        self.target_std = self.manifest["target_std_deg"]
        # Per-frame label validity. Dropouts are labels of exactly 0.0 recorded beside a
        # large value -- a logging default, not a centred wheel (3.9% of frames are exact
        # zeros; 973 of them sit next to a value above 5 deg, with a median first difference
        # of 69 deg against 0.10 overall). The image is identical to its neighbours' and the
        # target is wrong, so the frame is unlearnable and MSE would punish the model for a
        # corruption it cannot see. They stay IN the sequence, so windows remain contiguous
        # and the recurrent state is unbroken; they are masked out of the loss and metrics.
        dn = d / "dt_norm.npy"
        self.dt_norm = (torch.from_numpy(np.load(dn)).float() if dn.exists()
                        else torch.ones(len(self.angles_deg)))
        dp = d / "label_dropout.npy"
        self.label_valid = (torch.from_numpy(~np.load(dp)) if dp.exists()
                            else torch.ones(len(self.angles_deg), dtype=torch.bool))

    # ----------------------------------------------------------------- legal windows

    def window_starts(self, split, T):
        """Every legal start index for `split`, as a LongTensor."""
        lo, hi = self.manifest["splits"][split]
        out = []
        for a, b in self.manifest["segments"]:
            a, b = max(a, lo), min(b, hi)            # segment clipped to the split
            for s in range(a, b - T + 1):
                if self.missing and any(i in self.missing for i in range(s, s + T)):
                    continue
                out.append(s)
        return torch.tensor(out, dtype=torch.long)

    def windows_per_epoch(self, split, T):
        """The stride-4-equivalent count. Justify it in the README from the measured lag-1
        autocorrelation -- adjacent frames at the measured frame rate are nearly identical,
        so a stride-1 grid buys 4x the optimizer steps, not 4x the information."""
        return max(1, len(self.window_starts(split, T)) // TRAIN_STRIDE_EQUIVALENT)

    # ----------------------------------------------------------------- batch assembly

    def gather(self, starts, T):
        """(B,) start indices -> a Batch. Labels are in DEGREES here; the iterators standardise."""
        idx = starts[:, None] + torch.arange(T)
        f = self.frames[idx]                                  # (B,T,H,W,3) uint8, host
        f = f.to(self.device, non_blocking=True).permute(0, 1, 4, 2, 3).float().div_(255.0)
        y = self.angles_deg[idx].to(self.device, non_blocking=True)
        v = self.label_valid[idx].to(self.device, non_blocking=True)
        dt = self.dt_norm[idx].to(self.device, non_blocking=True)
        return Batch(f, y, v, dt)

    def standardise(self, y_deg):
        return (y_deg - self.target_mean) / self.target_std

    def to_degrees(self, y_std):
        return y_std * self.target_std + self.target_mean


# --------------------------------------------------------------------- augmentation

def center_crop(frames, model_w):
    off = (frames.shape[-1] - model_w) // 2
    return frames[..., off:off + model_w]


def augment_batch(frames, labels_deg, *, k_deg_per_px, model_w, generator=None,
                  brightness=0.25, shadow_prob=0.5, flip_prob=0.5, per_frame_bug=False):
    """Domain-adapted augmentation. frames (B,T,3,H,W) in [0,1]; labels (B,T) in DEGREES.

    Returns (frames (B,T,3,H,model_w), labels (B,T) in degrees).

    `per_frame_bug=True` samples every parameter per FRAME instead of per WINDOW. That is the
    deliberate control, not an option to use.

    Sign convention for the shift, stated because it cannot be derived here: this dataset
    publishes no camera calibration and no speed channel, and the steering ratio is
    undisclosed, so `k` is an ASSUMED coefficient resting on flat-ground, fixed-lookahead and
    constant-speed assumptions -- not a physical derivation. We take a crop window displaced
    `d` px to the right to stand in for the camera being displaced right, which the driver
    corrects by steering left, so the label becomes `a - k*d`. The hyperparameter sweep
    includes `k = 0`, so if the sign or the premise is wrong the curve has no interior
    minimum -- which is itself the result.
    """
    B, T, C, H, W = frames.shape
    dev, g = frames.device, generator

    # pT == 1 is the correct pipeline: one draw per window, broadcast over T.
    # pT == T is the deliberate bug. That single value is the entire difference, which is why
    # the bug-control runs share every other code path with the real runs.
    pT = T if per_frame_bug else 1

    def rand2():
        return torch.rand(B, pT, device=dev, generator=g)

    def rand5():
        return rand2().view(B, pT, 1, 1, 1)

    max_shift = (W - model_w) // 2
    start = torch.randint(0, 2 * max_shift + 1, (B, pT), device=dev, generator=g)
    d = (start - max_shift).float()

    col = start.view(B, pT, 1, 1, 1) + torch.arange(model_w, device=dev).view(1, 1, 1, 1, -1)
    out = frames.gather(4, col.expand(B, T, C, H, model_w))
    labels = labels_deg - k_deg_per_px * d          # (B,pT) broadcasts against (B,T)

    flip = rand2() < flip_prob
    out = torch.where(flip.view(B, pT, 1, 1, 1), out.flip(-1), out)
    labels = torch.where(flip, -labels, labels)

    out = (out * (1.0 + brightness * (2 * rand5() - 1))).clamp_(0.0, 1.0)

    # Shadow: one darkened vertical band per window. Photometric only -- it does NOT touch the
    # label, and it belongs in the SEEN-IN-TRAINING column of the OOD table.
    use = rand5() < shadow_prob
    x0 = rand5() * model_w
    x1 = x0 + (0.15 + 0.35 * rand5()) * model_w
    xs = torch.arange(model_w, device=dev).view(1, 1, 1, 1, -1).float()
    band = ((xs >= x0) & (xs < x1)) & use
    out = torch.where(band, out * (0.45 + 0.35 * rand5()), out)
    return out, labels


# ----------------------------------------------------------------------- iterators

def train_batches(data, T, batch_size, k_deg_per_px, epoch_seed, per_frame_bug=False,
                  photometric=True):
    """Random-start windows, augmented. Yields Batch with standardised labels.

    `photometric=False` disables flip, brightness and shadow so that `--aug none` means what
    it says. Without it "no augmentation" would still flip and re-light every window, and the
    ablation row would compare full augmentation against most of it.
    """
    g = torch.Generator(device="cpu").manual_seed(epoch_seed)
    starts = data.window_starts("train", T)
    n = data.windows_per_epoch("train", T)
    pick = starts[torch.randint(len(starts), (n,), generator=g)]
    gdev = torch.Generator(device=data.device).manual_seed(epoch_seed)
    for i in range(0, n - batch_size + 1, batch_size):
        b = data.gather(pick[i:i + batch_size], T)
        off = {} if photometric else dict(brightness=0.0, shadow_prob=0.0, flip_prob=0.0)
        f, y = augment_batch(b.frames, b.y, k_deg_per_px=k_deg_per_px, model_w=data.model_w,
                             generator=gdev, per_frame_bug=per_frame_bug, **off)
        yield Batch(f, data.standardise(y), b.valid, b.dt)


def window_batches(data, split, T, batch_size):
    """Deterministic, centre-cropped, NO augmentation. Used for per-epoch validation and for
    the per-position-in-window figure -- the state is reset at every window start, so each
    frame is scored at all T positions and the position comparison is within-frame."""
    starts = data.window_starts(split, T)
    for i in range(0, len(starts), batch_size):
        b = data.gather(starts[i:i + batch_size], T)
        yield Batch(center_crop(b.frames, data.model_w), data.standardise(b.y), b.valid, b.dt)


def rollout_chunks(data, split, chunk=256):
    """Contiguous stateful rollout over each (segment ∩ split), yielding chunks to carry `hx`
    between. This is the FINAL-NUMBER path: it covers every frame in the split exactly once,
    with no double counting and none discarded.

    Do not forward a whole 9,450-frame segment in one call -- that is ~1.5 GB of activations
    for the input tensor alone. Chunking with `hx` carried across is exactly equivalent, which
    `tests/test_dataset_windows.py::test_chunked_rollout_covers_every_frame_once` asserts.
    """
    lo, hi = data.manifest["splits"][split]
    for seg_lo, seg_hi in data.manifest["segments"]:
        seg_lo, seg_hi = max(seg_lo, lo), min(seg_hi, hi)
        if seg_hi <= seg_lo:
            continue
        for s in range(seg_lo, seg_hi, chunk):
            e = min(s + chunk, seg_hi)
            b = data.gather(torch.tensor([s]), e - s)
            yield (s, e), Batch(center_crop(b.frames, data.model_w),
                                data.standardise(b.y), b.valid, b.dt)
