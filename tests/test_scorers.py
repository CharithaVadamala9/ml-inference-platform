"""Tests for the RAGAS scoring LLM config (must stay decoupled from the judge)."""

from __future__ import annotations

from mlip.config import settings
from mlip.eval.scorers import _ragas_chat_model


def test_ragas_uses_ragas_model_not_judge_model(monkeypatch):
    # Simulate the judge being swapped to a non-Anthropic model. RAGAS must keep
    # using ragas_model — otherwise it would send an invalid model to the
    # Anthropic API and NaN every faithfulness/correctness score.
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr(settings, "judge_model", "llama3.2:1b")
    monkeypatch.setattr(settings, "ragas_model", "claude-haiku-4-5-20251001")

    llm = _ragas_chat_model()
    model = getattr(llm, "model", None) or getattr(llm, "model_name", None)
    assert model == "claude-haiku-4-5-20251001"
    assert model != settings.judge_model
