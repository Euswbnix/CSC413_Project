"""The evaluation metrics, in one place because training and evaluation must not disagree.

Checkpoint selection uses the same function that produces the reported number. Selecting on
global MSE and reporting per-bin MAE would pick the most shrunken checkpoint of every run --
exactly what binning exists to prevent -- so `train.py` imports `macro_skill` from here.

Three protocol decisions are baked in rather than left to the caller:

* **Bins are assigned by GROUND TRUTH, never by prediction.** Binning by prediction is
  degenerate in a way that flatters the model: a constant-zero predictor would show an empty
  curve bin and score undefined there instead of badly.
* **Metrics are per frame**, over all timesteps.
* **Invalid frames are excluded everywhere.** 973 frames carry a label of exactly 0.0 recorded
  beside a large value -- a logging default, not a centred wheel. They are unlearnable and
  they sit in the straight bin, so counting them would flatter predict-0 precisely where it
  is already strongest.

An independent TURN EVENT is a maximal run of in-bin frames, merging runs separated by fewer
than 15 frames. It travels with every count because 200 correlated frames from one corner are
one sample, not 200.
"""

import numpy as np

BIN_EDGES = [0.0, 5.0, 15.0, np.inf]
BIN_NAMES = ["straight [0,5)", "gentle [5,15)", "curve [15,inf)"]
DIAGNOSTIC_EDGE = 40.0
EVENT_MERGE_GAP = 15
FALSE_ALARM_EDGE = 5.0


def count_turn_events(mask, merge_gap=EVENT_MERGE_GAP):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return 0
    return 1 + int((np.diff(idx) > merge_gap).sum())


def bin_masks(true_deg, valid=None):
    """Ground-truth bins, intersected with validity. Returns [(name, mask), ...]."""
    a = np.abs(np.asarray(true_deg, dtype=np.float64))
    v = np.ones_like(a, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    out = [(nm, v & (a >= lo) & (a < hi))
           for nm, lo, hi in zip(BIN_NAMES, BIN_EDGES[:-1], BIN_EDGES[1:])]
    out.append((f"(diagnostic) >={DIAGNOSTIC_EDGE:.0f}", v & (a >= DIAGNOSTIC_EDGE)))
    return out


def mae(pred, true, mask):
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return float("nan")
    return float(np.abs(np.asarray(pred)[m] - np.asarray(true)[m]).mean())


def persistence(true_deg):
    """The naive baseline a reader will actually think of: yhat_t = y_{t-1}.

    Reported and then disposed of in one paragraph. The model receives images only and no
    past ground-truth angles, so persistence is not a solution to the posed task and is
    unavailable the moment labels are absent -- which is always, at deployment.
    """
    a = np.asarray(true_deg, dtype=np.float64)
    return np.concatenate([[a[0]], a[:-1]])


def per_bin(pred, true, valid=None):
    """Per-bin MAE for the model and both naive baselines, with support counts."""
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    zero = np.zeros_like(true)
    pers = persistence(true)
    rows = {}
    for name, m in bin_masks(true, valid):
        rows[name] = {
            "n_frames": int(m.sum()),
            "n_events": count_turn_events(m),
            "mean_abs_true": float(np.abs(true[m]).mean()) if m.any() else float("nan"),
            "mae_model": mae(pred, true, m),
            "mae_predict0": mae(zero, true, m),
            "mae_persistence": mae(pers, true, m),
            "p95_abs_err": (float(np.percentile(np.abs(pred[m] - true[m]), 95))
                            if m.any() else float("nan")),
        }
    return rows


def macro_skill(pred, true, valid=None):
    """1 - mean over the PRIMARY bins of (MAE_model / MAE_predict0).

    Predict-0 scores exactly 0 by construction and the score cannot be won by shrinking
    predictions toward the mean, which is why this is the headline scalar and the quantity
    checkpoint selection maximises. The diagnostic >=40 bin is excluded: with 2 independent
    turn events in the test split it is not a measurement, and letting it into the headline
    scalar would hand a sixth of that scalar to two corners.
    """
    rows = per_bin(pred, true, valid)
    ratios = [r["mae_model"] / r["mae_predict0"]
              for name, r in rows.items()
              if name in BIN_NAMES and r["n_frames"] > 0 and r["mae_predict0"] > 0]
    if not ratios:
        return float("nan")
    return float(1.0 - np.mean(ratios))


def pearson_r(pred, true, valid=None):
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    m = np.ones_like(t, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if m.sum() < 2:
        return float("nan")
    p, t = p[m] - p[m].mean(), t[m] - t[m].mean()
    d = np.sqrt((p * p).sum() * (t * t).sum())
    return float((p * t).sum() / d) if d > 0 else float("nan")


def false_alarm_rate(pred, true, valid=None):
    """P(|yhat| > 5 | |y| < 5). Predict-0 aces it by construction; report it anyway, and say so."""
    p, t = np.asarray(pred), np.asarray(true)
    m = np.ones_like(t, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    m = m & (np.abs(t) < FALSE_ALARM_EDGE)
    return float((np.abs(p[m]) > FALSE_ALARM_EDGE).mean()) if m.any() else float("nan")


def summary(pred, true, valid=None):
    return {
        "macro_skill": macro_skill(pred, true, valid),
        "global_mae": mae(pred, true, np.ones_like(np.asarray(true), dtype=bool)
                          if valid is None else np.asarray(valid, dtype=bool)),
        "pearson_r": pearson_r(pred, true, valid),
        "false_alarm_rate": false_alarm_rate(pred, true, valid),
        "n_valid": int(len(true) if valid is None else np.asarray(valid).sum()),
        "bins": per_bin(pred, true, valid),
    }


def mae_by_window_position(pred, true, valid, T):
    """Per-frame MAE as a function of position within the evaluation window, t = 1..T.

    The headline figure. Inputs are (n_windows, T). A length-invariant arm must come out
    FLAT: that is the figure's own self-test, and a non-flat line for the single-frame CNN
    means the position accounting is broken, not that the CNN has memory.
    """
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, T)
    true = np.asarray(true, dtype=np.float64).reshape(-1, T)
    val = np.asarray(valid, dtype=bool).reshape(-1, T)
    return np.array([mae(pred[:, t], true[:, t], val[:, t]) for t in range(T)])
