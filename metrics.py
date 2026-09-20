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

The bins are [0,5), [5,15) and [15,inf). There is no separate sharp bin: gate G1 measured the
recording and no chronological split can support one, because turns above 40 deg are
concentrated in the middle of the recording and the final 30% holds almost none, leaving the
test split 2-4 independent events under every candidate cut. `>=40` is therefore reported as
a SUBSET VIEW of the curve bin -- scored inside it, listed separately for diagnosis, never a
headline number of its own.
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
    checkpoint selection maximises.

    The mean runs over the THREE primary bins only. The `>=40` row is a SUBSET VIEW of the
    curve bin, not a fourth bin: those frames are real data with real errors and they are
    scored inside `curve [15,inf)` like any other. What is withheld from them is a headline
    number of their own -- with 2 independent turn events in the test split, a standalone
    sharp-bin MAE is not a measurement, and averaging it in as a fourth term would hand a
    quarter of the scalar to two corners.
    """
    rows = per_bin(pred, true, valid)
    use = [r for name, r in rows.items()
           if name in BIN_NAMES and r["n_frames"] > 0 and r["mae_predict0"] > 0]
    if not use:
        return float("nan")
    # AVERAGE THE MAEs, THEN TAKE THE RATIO -- not the mean of per-bin ratios.
    # Averaging ratios looks equivalent and is not: predict-0's MAE is 2.33 deg in the
    # straight bin and 38 deg in the curve bin, so a per-bin ratio divides by a denominator
    # sixteen times smaller on the easy bin. One degree of straight-bin error would then cost
    # sixteen times the score of one degree of curve-bin error, and the headline scalar would
    # be dominated by the easiest bin -- the exact opposite of why the project bins at all.
    # Measured: under the ratio-mean, models that beat predict-0 on curves (28.7 vs 38.0)
    # still scored -0.46 to -0.94 overall.
    return float(1.0 - np.mean([r["mae_model"] for r in use])
                 / np.mean([r["mae_predict0"] for r in use]))


def macro_mae(pred, true, valid=None, which="mae_model"):
    """Mean of the per-bin MAEs, in degrees. The interpretable form of the same quantity:
    `macro_skill = 1 - macro_mae(model) / macro_mae(predict-0)`. Report both -- the degrees
    are readable and the skill is comparable across splits."""
    rows = per_bin(pred, true, valid)
    vals = [r[which] for name, r in rows.items() if name in BIN_NAMES and r["n_frames"] > 0]
    return float(np.mean(vals)) if vals else float("nan")


def pearson_r(pred, true, valid=None):
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    m = np.ones_like(t, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if m.sum() < 2:
        return float("nan")
    p, t = p[m] - p[m].mean(), t[m] - t[m].mean()
    d = np.sqrt((p * p).sum() * (t * t).sum())
    # A constant predictor has zero variance, so r is genuinely undefined -- but that is a
    # RESULT, not a gap in the data. Collapse to the constant is this task's dominant failure
    # mode (measured: the LSTM lands there at 1e-3, 3e-3 and 1e-2 across three seeds, and the
    # CfC does too at 3e-3), so a bare NaN in a table reads as "missing" when it means
    # "the model predicts one number". `constant_prediction` below makes it reportable.
    return float((p * t).sum() / d) if d > 0 else float("nan")


def constant_prediction(pred, valid=None, tol=1e-6):
    """True when the model emits essentially one number regardless of input."""
    p = np.asarray(pred, dtype=np.float64)
    m = np.ones_like(p, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    return bool(m.sum() > 1 and float(p[m].std()) < tol)


def false_alarm_rate(pred, true, valid=None):
    """P(|yhat| > 5 | |y| < 5). Predict-0 aces it by construction; report it anyway, and say so."""
    p, t = np.asarray(pred), np.asarray(true)
    m = np.ones_like(t, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    m = m & (np.abs(t) < FALSE_ALARM_EDGE)
    return float((np.abs(p[m]) > FALSE_ALARM_EDGE).mean()) if m.any() else float("nan")


COLLAPSE_R = 0.05          # |Pearson r| below this is no better than no correlation
COLLAPSE_CURVE = 0.95      # curve-bin MAE within 5% of predict-0 is the baseline, not a model


def collapsed(s):
    """Did this run collapse to a constant predictor?

    The dominant failure mode on this task, and it has to be a first-class outcome rather
    than something read off a NaN. Measured across ten seeds: a collapsed run emits one
    number regardless of input, so its false-alarm rate is exactly 0.000, its Pearson r is
    undefined or indistinguishable from zero, and its curve-bin MAE sits within a degree of
    the constant-zero baseline.

    Reporting a median over a mixture of escaped and collapsed runs hides exactly the thing
    that matters -- a set of seeds that lands 2 escaped / 8 collapsed has a median that
    describes neither -- so the escape RATE is reported alongside metrics conditioned on
    outcome. Takes a summary() dict.
    """
    if s.get("constant_prediction"):
        return True
    r = s.get("pearson_r")
    bins = s.get("bins") or {}
    curve = bins.get("curve [15,inf)", {})
    m, base = curve.get("mae_model"), curve.get("mae_predict0")
    near_baseline = bool(m and base and m > COLLAPSE_CURVE * base)
    no_signal = (r is None) or (r != r) or (abs(r) < COLLAPSE_R)
    return bool(no_signal and near_baseline)


def summary(pred, true, valid=None):
    return {
        "macro_skill": macro_skill(pred, true, valid),
        "macro_mae": macro_mae(pred, true, valid),
        "macro_mae_predict0": macro_mae(pred, true, valid, which="mae_predict0"),
        "global_mae": mae(pred, true, np.ones_like(np.asarray(true), dtype=bool)
                          if valid is None else np.asarray(valid, dtype=bool)),
        "pearson_r": pearson_r(pred, true, valid),
        "constant_prediction": constant_prediction(pred, valid),
        "pred_std": float(np.asarray(pred)[np.ones_like(np.asarray(pred), dtype=bool)
                          if valid is None else np.asarray(valid, dtype=bool)].std()),
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
