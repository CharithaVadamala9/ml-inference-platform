"""Tests for the PR-comment markdown rendering."""

from __future__ import annotations

from mlip.eval.comment import render_gate_markdown, render_naive_markdown
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


def test_gate_markdown_shows_both_and_binding_flips():
    # Headline case: a noisy drop that the naive mean-threshold flags but the
    # statistical gate clears (within noise). The binding verdict flips with --naive.
    champ_pq = _pq([0.8] * 12, [0.7] * 12, [5] * 12)
    cand_pq = _pq([0.4, 1.0] * 6, [0.7] * 12, [5] * 12)  # mean -0.1, high variance
    stat = evaluate_gate_statistical(cand_pq, champ_pq)
    naive = evaluate_gate(
        {"faithfulness": 0.7, "answer_correctness": 0.7},
        {"scorecard": {"faithfulness": 0.8, "answer_correctness": 0.7}},
        tolerance=0.03,
    )
    assert stat.passed  # statistical: within noise
    assert not naive.passed  # naive: mean dropped below floor

    md_stat = render_gate_markdown(stat=stat, naive=naive, naive_binding=False)
    assert md_stat.splitlines()[0].endswith("✅ PASS")  # statistical binding
    assert "### Statistical" in md_stat and "### Naive" in md_stat

    md_naive = render_gate_markdown(stat=stat, naive=naive, naive_binding=True)
    assert md_naive.splitlines()[0].endswith("❌ FAIL")  # naive binding flips it


def test_naive_markdown():
    naive = evaluate_gate(
        {"faithfulness": 0.70, "answer_correctness": 0.70},
        {"scorecard": {"faithfulness": 0.75, "answer_correctness": 0.70}},
        tolerance=0.03,
    )
    md = render_naive_markdown(naive, tolerance=0.03)
    assert "naive" in md.lower()
    assert "`faithfulness`" in md
