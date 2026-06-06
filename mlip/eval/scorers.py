"""Scorers used by the eval pipeline.

Two complementary signals:
- RAGAS provides reference metrics: `faithfulness` (are the answer's claims
  grounded in the retrieved context?) and `answer_correctness` (does it match
  the ground truth?).
- An LLM-as-a-judge gives a holistic helpfulness rating with a short rationale,
  which catches issues the reference metrics miss.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mlip.config import settings
from mlip.serving.backends import get_backend

# ---- record shape shared across the pipeline --------------------------------

Record = dict[str, Any]  # {id, question, answer, contexts: list[str], ground_truth}


# ---- RAGAS reference metrics ------------------------------------------------


def _ragas_llm_and_embeddings():
    """Wrap our Claude judge + local embeddings for RAGAS."""
    from langchain_anthropic import ChatAnthropic
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set; add it to .env to run RAGAS.")

    llm = ChatAnthropic(
        model=settings.judge_model,
        api_key=settings.anthropic_api_key,
        temperature=0.0,
        max_tokens=1024,
    )
    embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def score_ragas(records: list[Record]) -> dict[str, Any]:
    """Run RAGAS over the records.

    Faithfulness needs retrieved context, so it is only computed when context is
    present (RAG mode). In no-RAG mode we score answer_correctness only.
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import answer_correctness, faithfulness

    has_context = any(r["contexts"] for r in records)
    metrics = [faithfulness, answer_correctness] if has_context else [answer_correctness]

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        )
        for r in records
    ]
    llm, embeddings = _ragas_llm_and_embeddings()
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        show_progress=False,
    )
    df = result.to_pandas()

    out: dict[str, Any] = {"answer_correctness": float(df["answer_correctness"].mean())}
    per_item = [
        {"answer_correctness": _safe(row.get("answer_correctness"))}
        for row in df.to_dict(orient="records")
    ]
    if has_context:
        out["faithfulness"] = float(df["faithfulness"].mean())
        for item, row in zip(per_item, df.to_dict(orient="records"), strict=False):
            item["faithfulness"] = _safe(row.get("faithfulness"))
    out["per_item"] = per_item
    return out


def _safe(value: Any) -> float | None:
    try:
        f = float(value)
        return f if f == f else None  # filter NaN
    except (TypeError, ValueError):
        return None


# ---- LLM-as-a-judge ---------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a strict evaluator of question-answering systems. Given a question, "
    "a reference answer, and a candidate answer, rate the candidate's overall "
    "quality (correctness + helpfulness) on an integer scale from 1 (poor) to 5 "
    "(excellent). Respond ONLY with compact JSON of the form "
    '{"score": <1-5>, "reason": "<one short sentence>"}.'
)

JUDGE_TEMPLATE = (
    "Question:\n{question}\n\nReference answer:\n{ground_truth}\n\n"
    "Candidate answer:\n{answer}\n\nReturn the JSON verdict now."
)


@dataclass
class JudgeVerdict:
    score: float  # normalized to 0-1
    raw_score: int
    reason: str


class LLMJudge:
    """Holistic LLM-as-a-judge using the configured judge provider (Claude)."""

    def __init__(self) -> None:
        self.backend = get_backend(settings.judge_provider, model=settings.judge_model)

    def score_one(self, record: Record) -> JudgeVerdict:
        prompt = JUDGE_TEMPLATE.format(
            question=record["question"],
            ground_truth=record["ground_truth"],
            answer=record["answer"],
        )
        reply = self.backend.chat(
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        return self._parse(reply)

    @staticmethod
    def _parse(reply: str) -> JudgeVerdict:
        try:
            start, end = reply.index("{"), reply.rindex("}") + 1
            data = json.loads(reply[start:end])
            raw = int(data["score"])
            raw = max(1, min(5, raw))
            return JudgeVerdict(
                score=(raw - 1) / 4.0, raw_score=raw, reason=str(data.get("reason", ""))
            )
        except (ValueError, KeyError, json.JSONDecodeError):
            # A judge that can't be parsed is treated as a mid score, flagged in the reason.
            return JudgeVerdict(
                score=0.5, raw_score=3, reason=f"unparseable judge reply: {reply[:80]!r}"
            )

    def score(self, records: list[Record]) -> dict[str, Any]:
        verdicts = [self.score_one(r) for r in records]
        mean = sum(v.score for v in verdicts) / len(verdicts) if verdicts else 0.0
        return {
            "judge_helpfulness": mean,
            "per_item": [
                {"score": v.score, "raw_score": v.raw_score, "reason": v.reason} for v in verdicts
            ],
        }
