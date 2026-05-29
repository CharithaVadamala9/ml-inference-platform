"""The quality gate.

Compares a candidate scorecard against the committed champion and fails if any
key metric regresses beyond a tolerance. This is the logic the GitHub Actions
workflow runs to block a PR when quality drops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Metrics the gate enforces. judge_helpfulness is informational (not gated)
# because it is the noisiest signal.
GATE_METRICS = ["faithfulness", "answer_correctness"]
DEFAULT_TOLERANCE = 0.03  # absorb LLM/embedding nondeterminism


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
    checks = [
        MetricCheck(m, float(candidate[m]), float(champion_scores[m]), tolerance)
        for m in (metrics or GATE_METRICS)
    ]
    return GateResult(checks)
