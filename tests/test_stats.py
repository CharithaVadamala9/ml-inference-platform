"""Tests for the statistical primitives behind the gate."""

from __future__ import annotations

from mlip.eval.stats import correct_pvalues, mcnemar, paired_bootstrap_ci


def test_bootstrap_ci_detects_clear_regression():
    # Every paired delta is a solid drop -> CI sits entirely below zero.
    ci = paired_bootstrap_ci([-0.3] * 15, seed=1)
    assert ci.hi < 0
    assert abs(ci.point - (-0.3)) < 1e-9
    assert ci.p_value == 0.0


def test_bootstrap_ci_passes_within_noise():
    # Deltas centered on zero -> CI straddles zero (not significant).
    ci = paired_bootstrap_ci([-0.5, 0.5] * 10, seed=1)
    assert ci.lo < 0 < ci.hi
    assert ci.p_value > 0.05


def test_bootstrap_is_reproducible_with_seed():
    deltas = [0.1, -0.2, 0.05, -0.3, 0.0, 0.2, -0.1, 0.15]
    a = paired_bootstrap_ci(deltas, seed=42)
    b = paired_bootstrap_ci(deltas, seed=42)
    assert (a.lo, a.hi, a.p_value) == (b.lo, b.hi, b.p_value)


def test_mcnemar_flags_discordant_regression():
    champ = [True] * 20
    cand = [False] * 8 + [True] * 12  # 8 regressions, 0 improvements
    res = mcnemar(champ, cand)
    assert res.b == 8 and res.c == 0
    assert res.p_value < 0.05


def test_mcnemar_symmetric_is_not_significant():
    champ = [True] * 5 + [False] * 5
    cand = [False] * 5 + [True] * 5  # 5 each way -> symmetric
    res = mcnemar(champ, cand)
    assert res.b == 5 and res.c == 5
    assert res.p_value == 1.0


def test_correction_reduces_false_positives():
    # One true signal + three null comparisons that crossed 0.05 by chance + nulls.
    pvals = [0.001, 0.04, 0.045, 0.048, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    naive = sum(1 for p in pvals if p < 0.05)  # would reject 4 (3 false positives)

    bh = sum(correct_pvalues(pvals, alpha=0.05, method="benjamini-hochberg"))
    bonf = sum(correct_pvalues(pvals, alpha=0.05, method="bonferroni"))

    assert naive == 4
    assert bh < naive  # correction removes the chance crossings
    assert bh == 1  # only the genuine signal survives
    assert bonf <= bh


def test_correction_empty_list():
    assert correct_pvalues([], alpha=0.05) == []
