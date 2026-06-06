"""The quality gate.

Compares a candidate scorecard against the committed champion and fails if any
key metric regresses beyond a tolerance. This is the logic the GitHub Actions
workflow runs to block a PR when quality drops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlip.config import settings
from mlip.eval.stats import correct_pvalues, mcnemar, paired_bootstrap_ci

# Metrics the gate enforces. judge is informational by default (settings.gate_judge).
GATE_METRICS = ["faithfulness", "answer_correctness"]
DEFAULT_TOLERANCE = 0.03  # absorb LLM/embedding nondeterminism (naive mode only)


@dataclass
class MetricCheck:
    metric: str
    candidate: float
    champion: float
    tolerance: float

    @property
    def floor(self) -> float:
        return self.champion - self.tolerance

    @property
    def passed(self) -> bool:
        return self.candidate >= self.floor


@dataclass
class GateResult:
    checks: list[MetricCheck]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[MetricCheck]:
        return [c for c in self.checks if not c.passed]


def evaluate_gate(
    candidate: dict[str, float],
    champion_record: dict[str, Any],
    *,
    metrics: list[str] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> GateResult:
    champion_scores = champion_record["scorecard"]
    # Only gate on metrics present in both the candidate and the champion
    # (a no-RAG run has no faithfulness to gate on).
    gate_metrics = [m for m in (metrics or GATE_METRICS) if m in candidate and m in champion_scores]
    checks = [
        MetricCheck(m, float(candidate[m]), float(champion_scores[m]), tolerance)
        for m in gate_metrics
    ]
    return GateResult(checks)


# ---------------------------------------------------------------------------
# Statistical gate (paired bootstrap CI + McNemar + multiple-comparison control)
# ---------------------------------------------------------------------------

PerQuestion = list[dict[str, Any]]


@dataclass
class StatTest:
    metric: str
    category: str
    n: int
    champion_mean: float
    candidate_mean: float
    delta: float  # candidate - champion (negative = regression)
    ci_lo: float
    ci_hi: float
    p_value: float  # per-comparison two-sided bootstrap/McNemar p (NOT multiplicity-adjusted)
    kind: str  # "bootstrap" | "mcnemar"
    gated: bool  # can this test fail the build?
    corrected_reject: bool = False  # BH/Bonferroni-adjusted p < alpha (the binding signal)
    b: int | None = None  # McNemar: champion-pass/candidate-fail
    c: int | None = None  # McNemar: champion-fail/candidate-pass

    @property
    def regression(self) -> bool:
        """Binding verdict: a gated regression whose multiplicity-ADJUSTED p < alpha.

        The displayed ci_lo/ci_hi are per-comparison (not multiplicity-adjusted);
        the gate decision uses the corrected p-value to stay coherent with the
        multiple-comparison control.
        """
        return self.gated and self.delta < 0 and self.corrected_reject


@dataclass
class StatGateResult:
    tests: list[StatTest]
    matched: int
    dropped_candidate: int
    dropped_champion: int
    content_mismatches: int
    correction_method: str
    alpha: float
    insufficient_overlap: bool
    min_paired: int

    @property
    def gated_failures(self) -> list[StatTest]:
        return [t for t in self.tests if t.regression]

    @property
    def passed(self) -> bool:
        if self.insufficient_overlap or self.content_mismatches > 0:
            return False
        return not self.gated_failures


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _categories(valid_ids: list[str], pq: dict[str, dict]) -> dict[str, list[str]]:
    """Group ids by category, plus an 'overall' bucket spanning everything."""
    groups: dict[str, list[str]] = {"overall": list(valid_ids)}
    for i in valid_ids:
        cat = pq[i].get("category", "uncategorized")
        if cat and cat != "overall":
            groups.setdefault(cat, []).append(i)
    # With only the default 'uncategorized' tag, return just the overall bucket.
    if set(groups) == {"overall", "uncategorized"} and len(groups["uncategorized"]) == len(
        valid_ids
    ):
        return {"overall": list(valid_ids)}
    return groups


def evaluate_gate_statistical(
    candidate_pq: PerQuestion,
    champion_pq: PerQuestion,
    *,
    gate_metrics: list[str] | None = None,
    alpha: float | None = None,
    n_resamples: int | None = None,
    seed: int | None = None,
    correction_method: str | None = None,
    judge_pass_threshold: int | None = None,
    gate_judge: bool | None = None,
    min_paired: int | None = None,
) -> StatGateResult:
    """Paired, statistically-honest gate over per-question scores.

    Joins candidate<->champion on `id` (verifying content_hash), runs a paired
    bootstrap CI per continuous metric and McNemar for the binary judge metric,
    then applies a multiple-comparison correction across the gated tests.

    Stratified gating: when questions carry categories, a test is created for each
    gated metric on the aggregate ("overall") AND on each category. ALL of these
    gated tests form a SINGLE multiple-comparison family, so the gate catches both
    a diffuse aggregate regression and one concentrated in a single subgroup
    (e.g. "unanswerable") even when the aggregate looks fine.

    Power caveat (intended, not a flaw): per-category n is small, so only sizeable
    category regressions are detectable; subtle subgroup drops can wash out. Add
    more questions per category to raise sensitivity.
    """
    alpha = settings.gate_alpha if alpha is None else alpha
    n_resamples = settings.bootstrap_resamples if n_resamples is None else n_resamples
    seed = settings.bootstrap_seed if seed is None else seed
    correction_method = (
        settings.correction_method if correction_method is None else correction_method
    )
    judge_pass_threshold = (
        settings.judge_pass_threshold if judge_pass_threshold is None else judge_pass_threshold
    )
    gate_judge = settings.gate_judge if gate_judge is None else gate_judge
    min_paired = settings.min_paired_questions if min_paired is None else min_paired
    gate_metrics = gate_metrics or GATE_METRICS

    cand = {r["id"]: r for r in candidate_pq}
    champ = {r["id"]: r for r in champion_pq}
    common = set(cand) & set(champ)

    def _mismatch(i: str) -> bool:
        ch, ph = cand[i].get("content_hash"), champ[i].get("content_hash")
        return bool(ch and ph and ch != ph)

    content_mismatches = sum(1 for i in common if _mismatch(i))
    valid_ids = sorted(i for i in common if not _mismatch(i))
    matched = len(valid_ids)
    result = StatGateResult(
        tests=[],
        matched=matched,
        dropped_candidate=len(set(cand) - set(champ)),
        dropped_champion=len(set(champ) - set(cand)),
        content_mismatches=content_mismatches,
        correction_method=correction_method,
        alpha=alpha,
        insufficient_overlap=matched < min_paired,
        min_paired=min_paired,
    )
    if result.insufficient_overlap:
        return result

    tests: list[StatTest] = []
    for cat, ids in _categories(valid_ids, cand).items():
        # continuous metrics -> paired bootstrap CI
        for metric in gate_metrics:
            deltas, cvals, pvals = [], [], []
            for i in ids:
                cv, pv = cand[i].get(metric), champ[i].get(metric)
                if cv is None or pv is None:
                    continue
                deltas.append(cv - pv)
                cvals.append(cv)
                pvals.append(pv)
            if not deltas:
                continue
            ci = paired_bootstrap_ci(deltas, n_resamples=n_resamples, alpha=alpha, seed=seed)
            tests.append(
                StatTest(
                    metric=metric,
                    category=cat,
                    n=len(deltas),
                    champion_mean=_mean(pvals),
                    candidate_mean=_mean(cvals),
                    delta=ci.point,
                    ci_lo=ci.lo,
                    ci_hi=ci.hi,
                    p_value=ci.p_value,
                    kind="bootstrap",
                    gated=True,
                )
            )

        # judge -> binary pass/fail -> McNemar (informational unless gate_judge)
        champ_pass, cand_pass = [], []
        for i in ids:
            cr, pr = cand[i].get("judge_raw"), champ[i].get("judge_raw")
            if cr is None or pr is None:
                continue
            cand_pass.append(cr >= judge_pass_threshold)
            champ_pass.append(pr >= judge_pass_threshold)
        if champ_pass:
            mc = mcnemar(champ_pass, cand_pass)
            cand_rate, champ_rate = (
                _mean([float(x) for x in cand_pass]),
                _mean([float(x) for x in champ_pass]),
            )
            tests.append(
                StatTest(
                    metric="judge_pass",
                    category=cat,
                    n=len(champ_pass),
                    champion_mean=champ_rate,
                    candidate_mean=cand_rate,
                    delta=cand_rate - champ_rate,
                    ci_lo=float("nan"),
                    ci_hi=float("nan"),
                    p_value=mc.p_value,
                    kind="mcnemar",
                    gated=gate_judge,
                    b=mc.b,
                    c=mc.c,
                )
            )

    # multiple-comparison correction across the GATED family only
    gated_tests = [t for t in tests if t.gated]
    rejected = correct_pvalues(
        [t.p_value for t in gated_tests], alpha=alpha, method=correction_method
    )
    for t, rej in zip(gated_tests, rejected, strict=True):
        t.corrected_reject = rej

    result.tests = tests
    return result
