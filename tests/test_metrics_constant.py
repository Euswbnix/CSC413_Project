#!/usr/bin/env python3
"""CCC and the best-constant baseline: each check has a partner that MUST fail on bad input.

Written must-fire first because this project's bugs have repeatedly been in verification
code that only ever saw good input and therefore never had a chance to fire.
"""
import pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import metrics as M

RNG = np.random.default_rng(0)


def labels(n=3000):
    """Heavy-tailed like the real target: mostly near zero, a curvy tail on both sides."""
    y = RNG.normal(0, 3, n)
    k = RNG.random(n) < 0.3
    y[k] = RNG.choice([-1, 1], k.sum()) * RNG.uniform(5, 120, k.sum())
    return y


# ----------------------------------------------------------------------------- CCC

def test_ccc_perfect_and_constant():
    y = labels()
    assert abs(M.ccc(y, y) - 1.0) < 1e-12
    assert M.ccc(np.full_like(y, 3.7), y) == 0.0          # a constant scores exactly 0, not NaN
    assert np.isnan(M.pearson_r(np.full_like(y, 3.7), y))  # ...which r cannot say


def test_ccc_punishes_scale_and_location_where_r_does_not():
    """Checked against the closed forms rather than a threshold. With mu, s2 the mean and
    population variance of y: for a*y the CCC is 2a*s2 / (a^2*s2 + s2 + (a-1)^2*mu^2) --
    scaling also moves the mean unless mu is 0 -- and for y+b it is 2*s2 / (2*s2 + b^2)."""
    y = labels()
    mu, s2 = y.mean(), y.var()
    scaled = lambda a: 2 * a * s2 / (a * a * s2 + s2 + (a - 1) ** 2 * mu * mu)
    cases = [(0.5 * y, scaled(0.5)),
             (2.0 * y, scaled(2.0)),
             (y + 2 * np.sqrt(s2), 2 * s2 / (2 * s2 + 4 * s2))]
    for bad, want in cases:
        assert abs(M.pearson_r(bad, y) - 1.0) < 1e-12     # r is blind to all three
        assert abs(M.ccc(bad, y) - want) < 1e-9, (M.ccc(bad, y), want)
        assert want < 0.85                                # MUST fire: ccc is not blind


def test_ccc_respects_validity_mask():
    y = labels(); p = y.copy()
    v = np.ones_like(y, dtype=bool); v[:100] = False
    p[:100] += 500.0                                      # garbage only on invalid frames
    assert abs(M.ccc(p, y, v) - 1.0) < 1e-12
    assert M.ccc(p, y) < 0.9                              # MUST fire without the mask


def test_ccc_both_constant():
    """Found by an adversarial probe: a constant prediction against constant labels at a
    DIFFERENT value has cov 0 and a positive denominator, so the CCC is 0, not NaN."""
    assert M.ccc([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) == 0.0
    assert np.isnan(M.ccc([2.0, 2.0, 2.0], [2.0, 2.0, 2.0]))


def test_empty_input_does_not_crash():
    """Also found by the probe: persistence() indexed a[0] and crashed summary() on an empty
    split. Every metric should come back NaN instead."""
    s = M.summary(np.array([]), np.array([]), ref_constant=1.0)
    assert np.isnan(s["macro_mae"]) and np.isnan(s["ccc"]) and s["n_valid"] == 0


# ---------------------------------------------------------------- best constant

def brute(y, v=None):
    cands = np.unique(np.asarray(y)[np.ones_like(y, bool) if v is None else v])
    return min(M.macro_mae(np.full_like(y, c), y, v) for c in cands)


def test_weights_reproduce_macro_mae():
    y = labels(); p = y + RNG.normal(0, 4, y.size)
    w = M.macro_weights(y)
    assert abs((w * np.abs(p - y)).sum() - M.macro_mae(p, y)) < 1e-9
    assert abs(w.sum() - 1.0) < 1e-12


def test_best_constant_is_the_exact_minimiser():
    for _ in range(5):
        y = labels(800)
        v = RNG.random(y.size) > 0.05
        c = M.best_constant(y, v)
        got = M.macro_mae(np.full_like(y, c), y, v)
        assert got <= brute(y, v) + 1e-9, (got, brute(y, v))


def test_plain_median_is_not_the_answer():
    """MUST fire: if best_constant quietly used the unweighted median, this would pass
    silently on symmetric data. Skewed labels make the two disagree."""
    y = np.concatenate([np.full(900, 0.5), np.full(50, 10.0), np.full(50, 60.0)])
    plain = float(np.median(y))
    c = M.best_constant(y)
    assert plain == 0.5
    assert c != plain
    assert M.macro_mae(np.full_like(y, c), y) < M.macro_mae(np.full_like(y, plain), y)


def test_skill_vs_constant_zero_point_and_predict0():
    y = labels(); c = M.best_constant(y)
    assert abs(M.macro_skill_vs_constant(np.full_like(y, c), y, None, c)) < 1e-12
    assert M.macro_skill_vs_constant(np.zeros_like(y), y, None, c) <= 1e-12


def test_selection_ranking_is_unchanged():
    """Skill against any fixed reference is monotone in macro MAE, so switching the
    reference must not reorder checkpoints."""
    y = labels(); c = M.best_constant(y)
    preds = [y + RNG.normal(0, s, y.size) for s in (1, 3, 9, 20)]
    a = np.argsort([M.macro_skill(p, y) for p in preds])
    b = np.argsort([M.macro_skill_vs_constant(p, y, None, c) for p in preds])
    assert (a == b).all()


def test_summary_keeps_old_fields_bit_identical():
    y = labels(); p = y + RNG.normal(0, 5, y.size)
    s = M.summary(p, y, ref_constant=1.0)
    assert s["macro_skill"] == M.macro_skill(p, y)
    assert s["macro_mae"] == M.macro_mae(p, y)
    assert s["pearson_r"] == M.pearson_r(p, y)
    for k in ("ccc", "oracle_constant", "macro_mae_oracle_constant", "ref_constant",
              "macro_mae_ref_constant", "macro_skill_vs_ref_constant"):
        assert k in s, k
    assert "ref_constant" not in M.summary(p, y)          # absent unless supplied


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f(); print(f"  ✓ {f.__name__}")
    print(f"\n  {len(fns)} passed")
