"""Tests for the A/B comparison logic and champion store (no network)."""

from __future__ import annotations

import mlip.eval.champion as champ
from mlip.eval.ab import TIE_EPSILON, MetricComparison, compare_scorecards


def test_metric_comparison_winner():
    assert MetricComparison("faithfulness", 0.6, 0.8).winner == "B"
    assert MetricComparison("faithfulness", 0.8, 0.6).winner == "A"
    assert MetricComparison("faithfulness", 0.80, 0.80 + TIE_EPSILON / 2).winner == "tie"


def test_compare_scorecards_covers_all_metrics():
    a = {"faithfulness": 0.6, "answer_correctness": 0.5, "judge_helpfulness": 0.7}
    b = {"faithfulness": 0.7, "answer_correctness": 0.5, "judge_helpfulness": 0.6}
    comps = compare_scorecards(a, b)
    by_metric = {c.metric: c for c in comps}
    assert by_metric["faithfulness"].winner == "B"
    assert by_metric["answer_correctness"].winner == "tie"
    assert by_metric["judge_helpfulness"].winner == "A"


def test_champion_promote_and_load_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(champ, "CHAMPION_PATH", tmp_path / "champion.json")
    assert champ.load_champion() is None
    rec = champ.promote(
        config={"name": "c", "prompt_version": "v2"},
        scorecard={"faithfulness": 0.7},
        mlflow_run_id=None,
        tag_mlflow=False,
    )
    loaded = champ.load_champion()
    assert loaded is not None
    assert loaded["scorecard"]["faithfulness"] == 0.7
    assert loaded["config"]["name"] == "c"
    assert "promoted_at" in rec
