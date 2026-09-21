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

import json
import pathlib

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
    # float64 on purpose. Predictions arrive as float32, and a float32 mean depends on the
    # SIMD summation order of the platform: the same saved predictions gave global MAE values
    # differing at ~1e-7 relative between x86 (the 5090 host) and ARM (a Mac). Harmless in
    # size, but a reported number should not depend on where it was computed.
    return float(np.abs(np.asarray(pred, dtype=np.float64)[m]
                        - np.asarray(true, dtype=np.float64)[m]).mean())


def persistence(true_deg):
    """The naive baseline a reader will actually think of: yhat_t = y_{t-1}.

    Reported and then disposed of in one paragraph. The model receives images only and no
    past ground-truth angles, so persistence is not a solution to the posed task and is
    unavailable the moment labels are absent -- which is always, at deployment.
    """
    a = np.asarray(true_deg, dtype=np.float64)
    if a.size == 0:
        return a
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
    """1 - (mean over the PRIMARY bins of MAE_model) / (mean over those bins of MAE_predict0).

    Note the order: the MAEs are averaged FIRST and the ratio taken once. This is not the
    mean of per-bin ratios -- see the comment on the return statement for why they differ.

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
    if np.ptp(p[m]) == 0 or np.ptp(t[m]) == 0:
        # Exactly constant on either side: r is undefined. Checked BEFORE centring, because
        # centring a constant leaves a rounding residue of ~1e-16 that makes the denominator
        # positive, and the function then returned a meaningless ~1e-13 instead of NaN --
        # or NaN, depending on whether the particular value happened to average exactly.
        return float("nan")
    p, t = p[m] - p[m].mean(), t[m] - t[m].mean()
    d = np.sqrt((p * p).sum() * (t * t).sum())
    # A constant predictor has zero variance, so r is genuinely undefined -- but that is a
    # RESULT, not a gap in the data. Collapse to the constant is this task's dominant failure
    # mode (measured: the LSTM lands there at 1e-3, 3e-3 and 1e-2 across three seeds, and the
    # CfC does too at 3e-3), so a bare NaN in a table reads as "missing" when it means
    # "the model predicts one number". `constant_prediction` below makes it reportable.
    return float((p * t).sum() / d) if d > 0 else float("nan")


def ccc(pred, true, valid=None):
    """Lin's concordance correlation coefficient: agreement, not just association.

    2*cov / (var_pred + var_true + (mean_pred - mean_true)^2). It is Pearson r multiplied by
    a penalty for any mismatch in scale or location, so it cannot be won by a model that
    tracks the signal at the wrong amplitude -- and, unlike r, a constant predictor scores
    exactly 0 instead of an undefined NaN. That second property is why it is here: on this
    task macro MAE is largely won by near-constant output (of the 22 LSTM seeds that beat
    predict-0 on test, 15 have |r| < 0.02), so the report needs a number a constant cannot
    earn. Population moments (ddof=0), as in Lin (1989).
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    m = np.ones_like(t, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if m.sum() < 2:
        return float("nan")
    p, t = p[m], t[m]
    if np.ptp(p) == 0:
        # Exactly constant. Computed the long way, mean(p) carries a rounding residue of
        # ~1e-16 and the result is ~1e-18 instead of 0 -- harmless, but the claim above is
        # "exactly 0", and a collapsed run should read as 0 in every table. cov is 0, so the
        # value is 0 whenever the denominator is positive -- including when the labels are
        # constant too but at a different value -- and undefined only when both sides are the
        # same constant.
        den = t.var() + (p[0] - t.mean()) ** 2
        return 0.0 if den > 0 else float("nan")
    mp, mt = p.mean(), t.mean()
    cov = ((p - mp) * (t - mt)).mean()
    den = p.var() + t.var() + (mp - mt) ** 2
    return float(2.0 * cov / den) if den > 0 else float("nan")


def macro_weights(true, valid=None):
    """Per-frame weights w with macro_mae(pred) == sum(w * |pred - true|).

    Each PRIMARY bin that has frames gets total weight 1/B, split evenly over its frames.
    The `>=40` diagnostic view is a subset of the curve bin and gets no weight of its own,
    exactly as in `macro_mae`.
    """
    t = np.asarray(true, dtype=np.float64)
    w = np.zeros_like(t)
    masks = [mk for nm, mk in bin_masks(t, valid) if nm in BIN_NAMES and mk.any()]
    for mk in masks:
        w[mk] = 1.0 / (len(masks) * mk.sum())
    return w


def best_constant(true, valid=None):
    """The single number that minimises macro MAE on these labels.

    macro MAE of a constant c is sum_i w_i |c - y_i| with the weights above: convex and
    piecewise linear in c, so its minimiser is the WEIGHTED median of the labels -- exact, no
    search. This is not the plain median: the straight bin holds most frames but only a third
    of the weight, so the plain median sits too close to zero.

    Why it exists: macro skill is referenced to predict-0, but predict-0 is not the best
    trivial model. A well-chosen constant scores about 12.97 test macro MAE against
    predict-0's 13.40, so "beats predict-0" is a bar a constant clears.
    """
    t = np.asarray(true, dtype=np.float64)
    w = macro_weights(t, valid)
    keep = w > 0
    if not keep.any():
        return float("nan")
    t, w = t[keep], w[keep]
    o = np.argsort(t, kind="mergesort")
    t, w = t[o], w[o]
    cw = np.cumsum(w)
    return float(t[int(np.searchsorted(cw, 0.5 * cw[-1]))])


def reference_constant(processed, split="val"):
    """`best_constant` fitted on one split's valid labels, read from the processed directory.

    Numpy only, and it touches three small files (angles_deg.npy, label_dropout.npy,
    manifest.json), so `evaluate.py` and the offline recompute use literally the same number.
    Fitted on VALIDATION by default: that is the split model selection already sees, so a
    constant chosen there is a trivial model that could genuinely have been deployed. It uses
    every valid label in the split's index range; the rollout additionally skips the handful
    of frames whose image file is missing, which cannot move a weighted median measurably.
    """
    d = pathlib.Path(processed)
    y = np.load(d / "angles_deg.npy").astype(np.float64)
    lo, hi = json.loads((d / "manifest.json").read_text())["splits"][split]
    dp = d / "label_dropout.npy"
    v = ~np.load(dp).astype(bool) if dp.exists() else np.ones(len(y), dtype=bool)
    return best_constant(y[lo:hi], v[lo:hi])


def macro_skill_vs_constant(pred, true, valid, c):
    """1 - macro_mae(model) / macro_mae(constant c). The constant scores exactly 0.

    For checkpoint SELECTION this changes nothing: against any fixed reference, skill is a
    monotone function of the model's macro MAE, so the ranking of checkpoints is identical to
    `macro_skill`. What changes is where zero sits -- which is the whole point for reporting.
    """
    base = macro_mae(np.full(np.shape(true), float(c)), true, valid)
    return float(1.0 - macro_mae(pred, true, valid) / base) if base > 0 else float("nan")


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


def summary(pred, true, valid=None, ref_constant=None):
    """All reported metrics for one split.

    `ref_constant` is the best constant FITTED ON VALIDATION (see `best_constant`); pass it
    when scoring any split so the skill is referenced to a trivial model that could actually
    have been chosen. `oracle_constant` is fitted on the scored split itself -- it has seen
    the answers, so it is an upper bound on what any constant can do, never a competitor.
    """
    c_oracle = best_constant(true, valid)
    full = lambda c: np.full(np.shape(true), float(c))
    out = {
        "macro_skill": macro_skill(pred, true, valid),
        "macro_mae": macro_mae(pred, true, valid),
        "macro_mae_predict0": macro_mae(pred, true, valid, which="mae_predict0"),
        "global_mae": mae(pred, true, np.ones_like(np.asarray(true), dtype=bool)
                          if valid is None else np.asarray(valid, dtype=bool)),
        "pearson_r": pearson_r(pred, true, valid),
        "constant_prediction": constant_prediction(pred, valid),
        "pred_std": float(np.asarray(pred, dtype=np.float64)[
            np.ones_like(np.asarray(pred), dtype=bool) if valid is None
            else np.asarray(valid, dtype=bool)].std()),
        "false_alarm_rate": false_alarm_rate(pred, true, valid),
        "n_valid": int(len(true) if valid is None else np.asarray(valid).sum()),
        "bins": per_bin(pred, true, valid),
        "ccc": ccc(pred, true, valid),
        "oracle_constant": c_oracle,
        "macro_mae_oracle_constant": macro_mae(full(c_oracle), true, valid),
    }
    if ref_constant is not None:
        out["ref_constant"] = float(ref_constant)
        out["macro_mae_ref_constant"] = macro_mae(full(ref_constant), true, valid)
        out["macro_skill_vs_ref_constant"] = macro_skill_vs_constant(pred, true, valid,
                                                                     ref_constant)
    return out


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
