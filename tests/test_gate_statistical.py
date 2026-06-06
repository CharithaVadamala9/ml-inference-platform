"""Tests for the statistical (paired) quality gate."""

from __future__ import annotations

from mlip.eval.gate import evaluate_gate_statistical


def _pq(ids, faith, ac, jr, *, hash_suffix=""):
    """Build a per_question list from parallel value lists."""
    return [
        {
            "id": ids[k],
            "content_hash": f"h{ids[k]}{hash_suffix}",
            "category": "overall",
            "faithfulness": faith[k],
            "answer_correctness": ac[k],
            "judge_raw": jr[k],
        }
        for k in range(len(ids))
    ]


IDS = [f"q{i}" for i in range(10)]


def test_clear_regression_fails():
    champ = _pq(IDS, [0.9] * 10, [0.7] * 10, [5] * 10)
    cand = _pq(IDS, [0.5] * 10, [0.7] * 10, [5] * 10)  # faithfulness tanks
    result = evaluate_gate_statistical(cand, champ)
    assert not result.passed
    failed = {(t.metric, t.category) for t in result.gated_failures}
    assert ("faithfulness", "overall") in failed
    assert ("answer_correctness", "overall") not in failed  # unchanged -> within noise


def test_within_noise_passes():
    champ = _pq(IDS, [0.9] * 10, [0.7] * 10, [5] * 10)
    cand = _pq(IDS, [0.95, 0.85] * 5, [0.72, 0.68] * 5, [5] * 10)  # jitter around champion
    result = evaluate_gate_statistical(cand, champ)
    assert result.passed
    assert result.matched == 10
    assert all(not t.regression for t in result.tests)


def test_judge_is_informational_by_default():
    # Judge collapses (all pass -> all fail) but judge is informational -> gate still passes.
    champ = _pq(IDS, [0.9] * 10, [0.7] * 10, [5] * 10)
    cand = _pq(IDS, [0.9] * 10, [0.7] * 10, [2] * 10)
    result = evaluate_gate_statistical(cand, champ)
    judge = next(t for t in result.tests if t.metric == "judge_pass")
    assert judge.gated is False
    assert judge.regression is False
    assert result.passed  # judge drop does not block


def test_judge_can_be_gated_when_enabled():
    champ = _pq(IDS, [0.9] * 10, [0.7] * 10, [5] * 10)
    cand = _pq(IDS, [0.9] * 10, [0.7] * 10, [2] * 10)
    result = evaluate_gate_statistical(cand, champ, gate_judge=True)
    judge = next(t for t in result.tests if t.metric == "judge_pass")
    assert judge.gated is True
    assert not result.passed  # now the judge collapse blocks


def test_insufficient_overlap_flagged():
    champ = _pq(IDS, [0.9] * 10, [0.7] * 10, [5] * 10)
    cand = _pq(IDS[:3], [0.9] * 3, [0.7] * 3, [5] * 3)  # only 3 shared ids
    result = evaluate_gate_statistical(cand, champ, min_paired=5)
    assert result.insufficient_overlap
    assert not result.passed
    assert result.matched == 3


def test_content_hash_mismatch_fails():
    champ = _pq(IDS, [0.9] * 10, [0.7] * 10, [5] * 10)
    cand = _pq(IDS, [0.9] * 10, [0.7] * 10, [5] * 10)
    # Two ids now refer to a different question than the champion did.
    cand[8]["content_hash"] = "WRONG8"
    cand[9]["content_hash"] = "WRONG9"
    result = evaluate_gate_statistical(cand, champ)
    assert result.content_mismatches == 2
    assert result.matched == 8  # mismatched ids are dropped from pairing
    assert not result.passed  # but the mismatch still fails the gate
