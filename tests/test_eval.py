"""Tests for the eval pipeline wiring.

The expensive parts (RAG generation, RAGAS, the LLM judge) are stubbed so the
graph orchestration and the runner's report-writing are tested offline.
"""

from __future__ import annotations

import json

import mlip.eval.graph as graph_mod
from mlip.eval.runner import run_eval
from mlip.rag.corpus import EvalExample
from mlip.rag.pipeline import RagConfig, RagResult


class _FakePipeline:
    def __init__(self, config):
        self.config = config

    def answer(self, question: str) -> RagResult:
        return RagResult(question=question, answer=f"answer to {question}", contexts=["ctx"])


def _patch_graph(monkeypatch):
    monkeypatch.setattr(graph_mod, "RagPipeline", _FakePipeline)
    monkeypatch.setattr(
        graph_mod,
        "load_eval",
        lambda: [
            EvalExample(id="q1", question="Q1?", ground_truth="A1", source_ids=["d1"]),
            EvalExample(id="q2", question="Q2?", ground_truth="A2", source_ids=["d2"]),
        ],
    )
    monkeypatch.setattr(
        graph_mod,
        "score_ragas",
        lambda records: {"faithfulness": 0.8, "answer_correctness": 0.7, "per_item": []},
    )

    class _FakeJudge:
        def score(self, records):
            return {"judge_helpfulness": 0.9, "per_item": []}

    monkeypatch.setattr(graph_mod, "LLMJudge", _FakeJudge)


def test_eval_graph_produces_scorecard(monkeypatch):
    _patch_graph(monkeypatch)
    final = graph_mod.build_eval_graph().invoke({"config": RagConfig().to_dict()})
    sc = final["scorecard"]
    assert sc["faithfulness"] == 0.8
    assert sc["answer_correctness"] == 0.7
    assert sc["judge_helpfulness"] == 0.9
    assert sc["n_examples"] == 2.0
    assert len(final["records"]) == 2


def test_runner_writes_report(monkeypatch, tmp_path):
    _patch_graph(monkeypatch)
    monkeypatch.setattr("mlip.eval.runner.REPORTS_DIR", tmp_path)
    run = run_eval(RagConfig(name="unit"), log_to_mlflow=False)
    assert run.report_path.exists()
    saved = json.loads(run.report_path.read_text())
    assert saved["scorecard"]["faithfulness"] == 0.8
    assert saved["config"]["name"] == "unit"
    assert len(saved["records"]) == 2
