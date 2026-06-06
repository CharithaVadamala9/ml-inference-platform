"""Tests for the quality-gate logic (pure, no network)."""

from __future__ import annotations

from mlip.eval.gate import evaluate_gate

CHAMPION = {
    "scorecard": {"faithfulness": 0.75, "answer_correctness": 0.70, "judge_helpfulness": 0.9}
}


def test_gate_passes_when_metrics_hold():
    candidate = {"faithfulness": 0.76, "answer_correctness": 0.71}
    result = evaluate_gate(candidate, CHAMPION, tolerance=0.03)
    assert result.passed
    assert result.failures == []


def test_gate_tolerance_absorbs_small_noise():
    # 0.72 is below champion 0.75 but within the 0.03 tolerance floor (0.72).
    candidate = {"faithfulness": 0.72, "answer_correctness": 0.70}
    assert evaluate_gate(candidate, CHAMPION, tolerance=0.03).passed


def test_gate_fails_on_regression():
    candidate = {"faithfulness": 0.60, "answer_correctness": 0.70}
    result = evaluate_gate(candidate, CHAMPION, tolerance=0.03)
    assert not result.passed
    assert [c.metric for c in result.failures] == ["faithfulness"]


def test_gate_skips_metric_missing_from_candidate():
    # A no-RAG candidate has no faithfulness; the gate should only check what exists.
    candidate = {"answer_correctness": 0.72}
    result = evaluate_gate(candidate, CHAMPION, tolerance=0.03)
    assert [c.metric for c in result.checks] == ["answer_correctness"]
    assert result.passed
