"""Tests for the eval judge disk cache (avoids re-calling the paid API)."""

from __future__ import annotations

from mlip.config import settings
from mlip.eval.judge_cache import JudgeCache
from mlip.eval.scorers import LLMJudge


def test_cache_roundtrip(tmp_path):
    cache = JudgeCache("test-model", enabled=True, cache_dir=str(tmp_path))
    assert cache.get("q", "a") is None
    cache.set("q", "a", {"score": 0.5, "raw_score": 3, "reason": "x"})
    got = cache.get("q", "a")
    assert got is not None and got["raw_score"] == 3
    # Different (question, answer) is a different key.
    assert cache.get("q", "different") is None


def test_disabled_cache_is_noop(tmp_path):
    cache = JudgeCache("m", enabled=False, cache_dir=str(tmp_path))
    cache.set("q", "a", {"score": 1.0, "raw_score": 5, "reason": "y"})
    assert cache.get("q", "a") is None


def test_judge_uses_cache_and_skips_second_call(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "judge_cache_dir", str(tmp_path))
    monkeypatch.setattr(settings, "judge_cache_enabled", True)

    judge = LLMJudge()

    calls = {"n": 0}

    class _CountingBackend:
        def chat(self, messages, *, temperature=0.0, max_tokens=512):
            calls["n"] += 1
            return '{"score": 4, "reason": "ok"}'

    judge.backend = _CountingBackend()
    record = {"question": "What is X?", "ground_truth": "X is Y.", "answer": "X is Y."}

    v1 = judge.score_one(record)
    v2 = judge.score_one(record)  # identical -> served from cache

    assert v1.raw_score == 4
    assert v2.raw_score == 4
    assert calls["n"] == 1  # the API was called only once
