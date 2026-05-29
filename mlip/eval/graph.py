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
    scorecard = {
        "faithfulness": state["ragas"]["faithfulness"],
        "answer_correctness": state["ragas"]["answer_correctness"],
        "judge_helpfulness": state["judge"]["judge_helpfulness"],
        "n_examples": float(len(state["records"])),
    }
    return {"scorecard": scorecard}


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
