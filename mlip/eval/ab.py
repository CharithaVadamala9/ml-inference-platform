"""A/B harness: run two RAG configs through the eval pipeline and compare.

All three metrics are 'higher is better', so a winner is decided per metric and
overall by the chosen primary metric. Both runs are logged to MLflow by the
runner, so the comparison is reproducible from the registry.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlip.eval.runner import EvalRun, run_eval
from mlip.rag.pipeline import RagConfig

COMPARE_METRICS = ["faithfulness", "answer_correctness", "judge_helpfulness"]
TIE_EPSILON = 0.005  # score differences smaller than this are treated as a tie


@dataclass
class MetricComparison:
    metric: str
    a: float
    b: float

    @property
    def delta(self) -> float:
        return self.b - self.a

    @property
    def winner(self) -> str:
        if abs(self.delta) < TIE_EPSILON:
            return "tie"
        return "B" if self.delta > 0 else "A"


@dataclass
class ABResult:
    run_a: EvalRun
    run_b: EvalRun
    comparisons: list[MetricComparison]
    primary_metric: str

    @property
    def overall_winner(self) -> str:
        for c in self.comparisons:
            if c.metric == self.primary_metric:
                return c.winner
        return "tie"


def compare_scorecards(
    a: dict[str, float], b: dict[str, float], primary_metric: str = "faithfulness"
) -> list[MetricComparison]:
    # Only compare metrics both runs actually have (a no-RAG run has no faithfulness).
    metrics = [m for m in COMPARE_METRICS if m in a and m in b]
    return [MetricComparison(m, float(a[m]), float(b[m])) for m in metrics]


def run_ab(
    config_a: RagConfig,
    config_b: RagConfig,
    *,
    primary_metric: str = "faithfulness",
    log_to_mlflow: bool = True,
) -> ABResult:
    run_a = run_eval(config_a, log_to_mlflow=log_to_mlflow)
    run_b = run_eval(config_b, log_to_mlflow=log_to_mlflow)
    comparisons = compare_scorecards(run_a.scorecard, run_b.scorecard, primary_metric)
    return ABResult(run_a, run_b, comparisons, primary_metric)
