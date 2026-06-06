"""Tests for the PR-comment markdown rendering."""

from __future__ import annotations

from mlip.eval.comment import render_naive_markdown, render_stat_markdown
from mlip.eval.gate import evaluate_gate, evaluate_gate_statistical

IDS = [f"q{i}" for i in range(10)]


def _pq(faith, ac, jr):
    return [
        {
            "id": IDS[k],
            "content_hash": f"h{IDS[k]}",
            "category": "overall",
            "faithfulness": faith[k],
            "answer_correctness": ac[k],
            "judge_raw": jr[k],
        }
        for k in range(len(IDS))
    ]


def test_stat_markdown_pass():
    champ = _pq([0.9] * 10, [0.7] * 10, [5] * 10)
    cand = _pq([0.9] * 10, [0.7] * 10, [5] * 10)
    md = render_stat_markdown(evaluate_gate_statistical(cand, champ))
    assert "✅ PASS" in md
    assert "| Metric | Category |" in md
    assert "`faithfulness`" in md
    assert "informational" in md  # the judge row is informational


def test_stat_markdown_regression():
    champ = _pq([0.9] * 10, [0.7] * 10, [5] * 10)
    cand = _pq([0.5] * 10, [0.7] * 10, [5] * 10)  # faithfulness regresses
    md = render_stat_markdown(evaluate_gate_statistical(cand, champ))
    assert "❌ FAIL" in md
    assert "significant regression" in md


def test_stat_markdown_blocked_reason():
    champ = _pq([0.9] * 10, [0.7] * 10, [5] * 10)
    cand = _pq([0.9] * 10, [0.7] * 10, [5] * 10)
    result = evaluate_gate_statistical(cand, champ)
    md = render_stat_markdown(result, blocked_reason="content mismatch")
    assert "❌ FAIL" in md
    assert "Blocked" in md and "content mismatch" in md


def test_naive_markdown():
    naive = evaluate_gate(
        {"faithfulness": 0.70, "answer_correctness": 0.70},
        {"scorecard": {"faithfulness": 0.75, "answer_correctness": 0.70}},
        tolerance=0.03,
    )
    md = render_naive_markdown(naive, tolerance=0.03)
    assert "naive" in md.lower()
    assert "`faithfulness`" in md
