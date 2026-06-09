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


def _pq_cat(specs):
    """Build per_question from (id, category, faithfulness, answer_correctness, judge_raw)."""
    return [
        {
            "id": i,
            "content_hash": f"h{i}",
            "category": cat,
            "faithfulness": f,
            "answer_correctness": a,
            "judge_raw": j,
        }
        for (i, cat, f, a, j) in specs
    ]


def test_concentrated_subgroup_regression_is_caught_even_if_aggregate_is_flat():
    # 30 questions across 3 categories. Two categories improve (+0.1), one
    # ("unanswerable") drops (-0.2). Aggregate faithfulness delta nets to ~0
    # (within noise), but the subgroup regression must still fail the gate.
    # Use faithfulness-gated categories (multi-hop carries the regression).
    champ_specs, cand_specs = [], []
    for k in range(10):
        champ_specs.append((f"f{k}", "factual", 0.8, 0.7, 5))
        cand_specs.append((f"f{k}", "factual", 0.9, 0.7, 5))  # +0.1
    for k in range(10):
        champ_specs.append((f"d{k}", "definition", 0.8, 0.7, 5))
        cand_specs.append((f"d{k}", "definition", 0.9, 0.7, 5))  # +0.1
    for k in range(10):
        champ_specs.append((f"m{k}", "multi-hop", 0.8, 0.7, 5))
        cand_specs.append((f"m{k}", "multi-hop", 0.6, 0.7, 5))  # -0.2

    result = evaluate_gate_statistical(_pq_cat(cand_specs), _pq_cat(champ_specs))
    failed = {(t.metric, t.category) for t in result.gated_failures}

    assert not result.passed
    assert ("faithfulness", "multi-hop") in failed  # subgroup caught
    assert ("faithfulness", "overall") not in failed  # aggregate stayed within noise


def test_small_subgroup_regression_within_noise_is_not_flagged():
    # Honest power limit: a small, noisy regression in a small subgroup washes out.
    champ_specs, cand_specs = [], []
    for k in range(10):  # a flat "common" category
        champ_specs.append((f"c{k}", "common", 0.8, 0.7, 5))
        cand_specs.append((f"c{k}", "common", 0.8, 0.7, 5))
    jitter = [0.70, 0.84, 0.70, 0.84, 0.70, 0.84, 0.70, 0.84]  # mean ~-0.03 vs 0.8, noisy
    for k in range(8):  # a small "rare" category with a tiny noisy drop
        champ_specs.append((f"r{k}", "rare", 0.8, 0.7, 5))
        cand_specs.append((f"r{k}", "rare", jitter[k], 0.7, 5))

    result = evaluate_gate_statistical(_pq_cat(cand_specs), _pq_cat(champ_specs))
    assert result.passed  # too small / too noisy to detect -> not flagged


def test_unanswerable_bucket_gated_on_correctness_only():
    # faithfulness is ill-defined for abstentions -> no faithfulness test for the
    # unanswerable bucket, and the aggregate faithfulness excludes those questions.
    champ = _pq_cat(
        [(f"f{k}", "factual", 0.8, 0.7, 5) for k in range(8)]
        + [(f"u{k}", "unanswerable", None, 0.9, 2) for k in range(8)]
    )
    cand = _pq_cat(
        [(f"f{k}", "factual", 0.8, 0.7, 5) for k in range(8)]
        + [(f"u{k}", "unanswerable", None, 0.9, 2) for k in range(8)]
    )
    result = evaluate_gate_statistical(cand, champ)
    by = {(t.metric, t.category) for t in result.tests}
    assert ("faithfulness", "unanswerable") not in by  # no faithfulness on abstentions
    assert ("answer_correctness", "unanswerable") in by  # correctness still gates it
    overall_faith = next(
        t for t in result.tests if t.metric == "faithfulness" and t.category == "overall"
    )
    assert overall_faith.n == 8  # aggregate faithfulness excludes the 8 unanswerable


def test_unanswerable_fabrication_caught_via_correctness():
    # Model stops abstaining and fabricates -> correctness on the unanswerable bucket tanks.
    champ = _pq_cat(
        [(f"u{k}", "unanswerable", None, 0.9, 2) for k in range(10)]
        + [(f"f{k}", "factual", 0.8, 0.8, 5) for k in range(10)]
    )
    cand = _pq_cat(
        [(f"u{k}", "unanswerable", None, 0.3, 2) for k in range(10)]
        + [(f"f{k}", "factual", 0.8, 0.8, 5) for k in range(10)]
    )
    result = evaluate_gate_statistical(cand, champ)
    failed = {(t.metric, t.category) for t in result.gated_failures}
    assert ("answer_correctness", "unanswerable") in failed
    assert not result.passed


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
