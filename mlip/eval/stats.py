"""Statistical primitives for the quality gate.

All functions are pure and deterministic given a seed, so gate verdicts are
reproducible. Used by `mlip/eval/gate.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    point: float  # observed mean delta
    lo: float  # lower bound of the (1 - alpha) CI
    hi: float  # upper bound
    p_value: float  # two-sided bootstrap p-value vs 0
    n: int


def paired_bootstrap_ci(
    deltas: list[float],
    *,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCI:
    """Bootstrap CI for the mean of paired deltas (candidate - champion).

    A regression is 'significant' when the whole CI is below zero (hi < 0).
    The two-sided p-value (fraction of resample means on the far side of 0) is
    what the multiple-comparison correction consumes.
    """
    arr = np.asarray(deltas, dtype=float)
    n = int(arr.size)
    if n == 0:
        return BootstrapCI(point=float("nan"), lo=float("nan"), hi=float("nan"), p_value=1.0, n=0)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = arr[idx].mean(axis=1)

    lo = float(np.percentile(boot_means, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    p_lo = float(np.mean(boot_means <= 0.0))
    p_hi = float(np.mean(boot_means >= 0.0))
    p_value = min(1.0, 2.0 * min(p_lo, p_hi))
    return BootstrapCI(point=float(arr.mean()), lo=lo, hi=hi, p_value=p_value, n=n)


@dataclass(frozen=True)
class McNemarResult:
    p_value: float
    b: int  # champion passed, candidate failed (regression direction)
    c: int  # champion failed, candidate passed (improvement direction)


def mcnemar(champion_pass: list[bool], candidate_pass: list[bool]) -> McNemarResult:
    """Exact McNemar test for paired binary outcomes via the binomial test."""
    b = sum(1 for ch, ca in zip(champion_pass, candidate_pass, strict=True) if ch and not ca)
    c = sum(1 for ch, ca in zip(champion_pass, candidate_pass, strict=True) if (not ch) and ca)
    n = b + c
    if n == 0:
        return McNemarResult(p_value=1.0, b=0, c=0)
    from scipy.stats import binomtest

    p = binomtest(b, n, 0.5, alternative="two-sided").pvalue
    return McNemarResult(p_value=float(p), b=b, c=c)


def correct_pvalues(
    pvalues: list[float], *, alpha: float = 0.05, method: str = "benjamini-hochberg"
) -> list[bool]:
    """Multiple-comparison correction. Returns, per test, whether it is rejected.

    Controls false positives across a family of tests:
    - 'bonferroni': reject if p <= alpha / m (strict FWER control)
    - 'benjamini-hochberg': step-up FDR control (default; more powerful)
    """
    if not pvalues:
        return []
    p = np.asarray(pvalues, dtype=float)
    m = p.size

    if method == "bonferroni":
        return [bool(x) for x in (p <= alpha / m)]

    if method != "benjamini-hochberg":
        raise ValueError(f"unknown correction method: {method!r}")

    order = np.argsort(p)
    ranked = p[order]
    thresholds = alpha * (np.arange(1, m + 1) / m)
    below = ranked <= thresholds
    rejected = np.zeros(m, dtype=bool)
    if below.any():
        kmax = int(np.max(np.where(below)[0]))
        rejected_sorted = np.zeros(m, dtype=bool)
        rejected_sorted[: kmax + 1] = True
        rejected[order] = rejected_sorted
    return [bool(x) for x in rejected]
