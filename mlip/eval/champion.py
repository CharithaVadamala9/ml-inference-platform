"""The 'champion' — the current best-known config and the quality bar to beat.

It is stored as a committed JSON file so the CI quality gate can read it without
needing access to the MLflow server, and (best-effort) tagged on the MLflow run
so the registry reflects which run is champion.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlip.config import settings

CHAMPION_PATH = Path(__file__).resolve().parents[2] / "data" / "champion.json"


def load_champion() -> dict[str, Any] | None:
    """Return the stored champion record, or None if none has been promoted."""
    if not CHAMPION_PATH.exists():
        return None
    return json.loads(CHAMPION_PATH.read_text(encoding="utf-8"))


def promote(
    *,
    config: dict[str, Any],
    scorecard: dict[str, float],
    per_question: list[dict[str, Any]] | None = None,
    mlflow_run_id: str | None = None,
    tag_mlflow: bool = True,
) -> dict[str, Any]:
    """Make this config the champion: write champion.json and tag the MLflow run.

    `per_question` (id-keyed scores) is stored so the statistical gate can run
    paired tests between a candidate and the champion on the same questions.
    """
    record = {
        "config": config,
        "scorecard": scorecard,
        "per_question": per_question or [],
        "mlflow_run_id": mlflow_run_id,
        "promoted_at": datetime.now(UTC).isoformat(),
    }
    CHAMPION_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if tag_mlflow and mlflow_run_id:
        _tag_champion(mlflow_run_id)
    return record


def _tag_champion(run_id: str) -> None:
    """Best-effort: mark this run as the champion in MLflow."""
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        client.set_tag(run_id, "alias", "champion")
    except Exception as exc:  # noqa: BLE001 - tagging is non-critical
        import warnings

        warnings.warn(f"MLflow champion tag skipped ({type(exc).__name__}: {exc})", stacklevel=2)
