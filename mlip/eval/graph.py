"""The agentic eval pipeline, expressed as a LangGraph state graph.

    generate -> score_ragas -> score_judge -> aggregate

Each node is a pure transformation of the shared state, which keeps the
pipeline inspectable and easy to extend (e.g. adding a retrieval-quality node
or fanning out scorers in parallel later).
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from mlip.eval.scorers import LLMJudge, score_ragas
from mlip.rag.corpus import load_eval
from mlip.rag.pipeline import RagConfig, RagPipeline


class EvalState(TypedDict, total=False):
    config: dict[str, Any]
    records: list[dict[str, Any]]
    ragas: dict[str, Any]
    judge: dict[str, Any]
    scorecard: dict[str, float]
    per_question: list[dict[str, Any]]


def _generate(state: EvalState) -> EvalState:
    config = RagConfig(**state["config"])
    pipeline = RagPipeline(config)
    examples = load_eval()
    records = []
    for ex in examples:
        result = pipeline.answer(ex.question)
        records.append(
            {
                "id": ex.id,
                # category is optional on EvalExample (added for stratified gating);
                # default keeps older datasets working.
                "category": getattr(ex, "category", "uncategorized"),
                "question": ex.question,
                "ground_truth": ex.ground_truth,
                "answer": result.answer,
                "contexts": result.contexts,
            }
        )
    return {"records": records}


def _score_ragas(state: EvalState) -> EvalState:
    return {"ragas": score_ragas(state["records"])}


def _score_judge(state: EvalState) -> EvalState:
    return {"judge": LLMJudge().score(state["records"])}


def _aggregate(state: EvalState) -> EvalState:
    ragas = state["ragas"]
    scorecard: dict[str, float] = {}
    # Faithfulness only exists in RAG mode (it needs retrieved context).
    if "faithfulness" in ragas:
        scorecard["faithfulness"] = ragas["faithfulness"]
    scorecard["answer_correctness"] = ragas["answer_correctness"]
    scorecard["judge_helpfulness"] = state["judge"]["judge_helpfulness"]
    scorecard["n_examples"] = float(len(state["records"]))
    return {"scorecard": scorecard, "per_question": _build_per_question(state)}


def _build_per_question(state: EvalState) -> list[dict[str, Any]]:
    """Merge per-item RAGAS + judge scores with the record id/category.

    This id-keyed structure is what the statistical gate reads to run paired
    tests between a candidate and the champion on the same questions.
    """
    records = state["records"]
    ragas_items = state["ragas"].get("per_item") or []
    judge_items = state["judge"].get("per_item") or []
    per_question = []
    for i, rec in enumerate(records):
        r = ragas_items[i] if i < len(ragas_items) else {}
        j = judge_items[i] if i < len(judge_items) else {}
        per_question.append(
            {
                "id": rec["id"],
                "category": rec.get("category", "uncategorized"),
                "faithfulness": r.get("faithfulness"),
                "answer_correctness": r.get("answer_correctness"),
                "judge_score": j.get("score"),
                "judge_raw": j.get("raw_score"),
                "judge_reason": j.get("reason"),
            }
        )
    return per_question


def build_eval_graph():
    """Compile and return the eval state graph."""
    g = StateGraph(EvalState)
    g.add_node("generate", _generate)
    g.add_node("score_ragas", _score_ragas)
    g.add_node("score_judge", _score_judge)
    g.add_node("aggregate", _aggregate)
    g.add_edge(START, "generate")
    g.add_edge("generate", "score_ragas")
    g.add_edge("score_ragas", "score_judge")
    g.add_edge("score_judge", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()
