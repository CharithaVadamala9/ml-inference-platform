"""Tests for the judge-calibration audit (Cohen's kappa + drift threshold)."""

from __future__ import annotations

from mlip.eval.calibration import kappa_audit, run_calibration
from mlip.eval.scorers import JudgeVerdict


def test_kappa_perfect_agreement_passes():
    r = kappa_audit([1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0], threshold=0.4)
    assert r.kappa == 1.0
    assert r.passed
    assert r.agreements == 6


def test_kappa_systematic_disagreement_fails():
    r = kappa_audit([1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1], threshold=0.4)
    assert r.kappa < 0.4
    assert not r.passed


def test_kappa_nan_handled():
    # both constant + identical -> undefined kappa -> treated as perfect agreement
    assert kappa_audit([1, 1, 1], [1, 1, 1], threshold=0.4).passed
    # judge constant but disagrees with a varied gold -> drift
    assert not kappa_audit([1, 0, 1, 0], [1, 1, 1, 1], threshold=0.4).passed


class _StubJudge:
    """Judge whose verdict is encoded in the answer text, so tests need no API."""

    def score_one(self, record):
        raw = 5 if record["answer"] == "GOOD" else 1
        return JudgeVerdict(score=(raw - 1) / 4.0, raw_score=raw, reason="")


def _items(pairs):
    return [
        {"question": f"q{i}", "ground_truth": "g", "answer": ans, "gold_pass": g}
        for i, (ans, g) in enumerate(pairs)
    ]


def test_run_calibration_agreement_passes(monkeypatch):
    monkeypatch.setattr("mlip.eval.scorers.LLMJudge", _StubJudge)
    items = _items([("GOOD", 1), ("BAD", 0), ("GOOD", 1), ("BAD", 0)])
    r = run_calibration(items=items, threshold=0.4)
    assert r.kappa == 1.0
    assert r.passed


def test_run_calibration_drift_fails(monkeypatch):
    monkeypatch.setattr("mlip.eval.scorers.LLMJudge", _StubJudge)
    # Judge rates everything a pass, but gold alternates -> no agreement -> drift.
    items = _items([("GOOD", 1), ("GOOD", 0), ("GOOD", 1), ("GOOD", 0)])
    r = run_calibration(items=items, threshold=0.4)
    assert not r.passed
