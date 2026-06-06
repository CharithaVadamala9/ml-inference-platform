"""Judge-calibration audit.

Measures whether the LLM judge still agrees with a small human-labeled gold set,
using Cohen's kappa. If agreement drops below a threshold the judge has drifted
and its verdicts (and therefore the champion baseline) are no longer trustworthy
— so the gate fails. Kappa is logged to MLflow so drift is visible over time.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from mlip.config import settings


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(__file__).resolve().parents[2] / p


@dataclass
class CalibrationResult:
    kappa: float
    n: int
    agreements: int
    threshold: float
    gold_labels: list[int]
    judge_labels: list[int]

    @property
    def passed(self) -> bool:
        return self.kappa >= self.threshold


def load_calibration(path: str | Path | None = None) -> list[dict]:
    fp = _resolve(path or settings.calibration_path)
    if not fp.exists():
        return []
    with fp.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def kappa_audit(
    gold_labels: list[int], judge_labels: list[int], *, threshold: float | None = None
) -> CalibrationResult:
    """Pure: Cohen's kappa between gold and judge binary labels + pass/fail."""
    from sklearn.metrics import cohen_kappa_score

    threshold = settings.kappa_threshold if threshold is None else threshold
    n = len(gold_labels)
    agreements = sum(1 for g, j in zip(gold_labels, judge_labels, strict=True) if g == j)
    if n == 0:
        kappa = 0.0
    else:
        kappa = float(cohen_kappa_score(gold_labels, judge_labels))
        if math.isnan(kappa):
            # Undefined (e.g. a rater is constant): perfect agreement -> 1, else 0.
            kappa = 1.0 if gold_labels == judge_labels else 0.0
    return CalibrationResult(
        kappa=kappa,
        n=n,
        agreements=agreements,
        threshold=threshold,
        gold_labels=gold_labels,
        judge_labels=judge_labels,
    )


def run_calibration(
    items: list[dict] | None = None, *, threshold: float | None = None
) -> CalibrationResult:
    """Run the judge over the calibration set and compare to gold labels."""
    from mlip.eval.scorers import LLMJudge

    items = load_calibration() if items is None else items
    judge = LLMJudge()
    gold_labels, judge_labels = [], []
    for it in items:
        verdict = judge.score_one(
            {"question": it["question"], "ground_truth": it["ground_truth"], "answer": it["answer"]}
        )
        judge_labels.append(1 if verdict.raw_score >= settings.judge_pass_threshold else 0)
        gold_labels.append(1 if it["gold_pass"] else 0)
    return kappa_audit(gold_labels, judge_labels, threshold=threshold)


def log_kappa_mlflow(kappa: float, *, experiment: str = "mlip-judge-calibration") -> None:
    """Best-effort: log kappa to MLflow so drift is visible over time."""
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name="calibration"):
            mlflow.log_param("judge_model", settings.judge_model)
            mlflow.log_metric("judge_kappa", kappa)
    except Exception as exc:  # noqa: BLE001 - logging must never break the audit
        import warnings

        warnings.warn(f"MLflow kappa logging skipped ({type(exc).__name__}: {exc})", stacklevel=2)
