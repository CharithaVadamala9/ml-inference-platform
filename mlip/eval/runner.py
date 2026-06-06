"""Run the eval graph for one RAG config, persist a report, log to MLflow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlip.config import settings
from mlip.eval.graph import build_eval_graph
from mlip.rag.pipeline import RagConfig

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
DEFAULT_EXPERIMENT = "mlip-rag-eval"


@dataclass
class EvalRun:
    config: dict[str, Any]
    scorecard: dict[str, float]
    records: list[dict[str, Any]]
    ragas: dict[str, Any]
    judge: dict[str, Any]
    report_path: Path
    per_question: list[dict[str, Any]] = field(default_factory=list)
    run_id: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def run_eval(
    config: RagConfig | None = None,
    *,
    experiment: str = DEFAULT_EXPERIMENT,
    log_to_mlflow: bool = True,
) -> EvalRun:
    config = config or RagConfig()
    final = build_eval_graph().invoke({"config": config.to_dict()})

    run = EvalRun(
        config=config.to_dict(),
        scorecard=final["scorecard"],
        records=final["records"],
        ragas={k: v for k, v in final["ragas"].items() if k != "per_item"},
        judge={k: v for k, v in final["judge"].items() if k != "per_item"},
        report_path=_report_path(config.name),
        per_question=final.get("per_question", []),
    )
    _write_report(run)
    if log_to_mlflow:
        run.run_id = _log_to_mlflow(run, experiment)
    return run


def _report_path(name: str) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPORTS_DIR / f"{name}-{stamp}.json"


def _write_report(run: EvalRun) -> None:
    payload = {
        "config": run.config,
        "scorecard": run.scorecard,
        # id-keyed per-question scores — what the statistical gate reads for paired tests.
        "per_question": run.per_question,
        "records": run.records,
        "started_at": run.started_at,
    }
    run.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _log_to_mlflow(run: EvalRun, experiment: str) -> str | None:
    """Log params/metrics/report to MLflow. Best-effort: never fail the eval."""
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run.config["name"]) as active:
            mlflow.log_params(run.config)
            mlflow.log_metrics(run.scorecard)
            mlflow.log_artifact(str(run.report_path))
            return active.info.run_id
    except Exception as exc:  # noqa: BLE001 - logging must never break the eval
        import warnings

        warnings.warn(f"MLflow logging skipped ({type(exc).__name__}: {exc})", stacklevel=2)
        return None
