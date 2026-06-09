"""The RAG question-answering system that the eval platform measures.

`RagConfig` captures every knob that defines a *variant* (prompt, retrieval
depth, model). Two configs with different knobs are exactly what the A/B harness
compares, so keeping them in one serializable object keeps experiments honest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from mlip.rag.corpus import Document
from mlip.rag.retriever import Retriever
from mlip.serving.backends import ChatBackend, get_backend

# Prompt variants are versioned so an A/B run records exactly which one produced
# a result. v2 tightens the grounding instruction to discourage hallucination.
PROMPT_VARIANTS: dict[str, str] = {
    "v1": (
        "You are a helpful assistant. Use the context below to answer the "
        "question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
    "v2": (
        "You are a helpful assistant. Use the context below and your own general "
        "knowledge to answer the question.\n\nContext:\n{context}\n\n"
        "Question: {question}\nAnswer:"
    ),
}

# Used in no-RAG mode: the model answers from its own knowledge, with no retrieved
# context. This lets the platform evaluate a bare model, not just a RAG pipeline.
DIRECT_PROMPT = "Answer the question concisely in 1-3 sentences.\n\nQuestion: {question}\nAnswer:"


@dataclass(frozen=True)
class RagConfig:
    name: str = "baseline"
    prompt_version: str = "v2"
    top_k: int = 3
    backend: str | None = None  # None -> configured default (ollama)
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 256
    use_retrieval: bool = True  # False -> bare model, no documents (no-RAG mode)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RagResult:
    question: str
    answer: str
    contexts: list[str]
    retrieved: list[Document] = field(default_factory=list)


class RagPipeline:
    def __init__(self, config: RagConfig | None = None, retriever: Retriever | None = None):
        self.config = config or RagConfig()
        # Only build the retriever when retrieval is actually used (no-RAG mode
        # needs no corpus or embedding model).
        if self.config.use_retrieval:
            self.retriever = retriever or Retriever()
        else:
            self.retriever = retriever
        self._backend: ChatBackend = get_backend(self.config.backend, model=self.config.model)

    def _format_context(self, docs: list[Document]) -> str:
        return "\n\n".join(f"[{d.title}] {d.text}" for d in docs)

    def answer(self, question: str) -> RagResult:
        if self.config.use_retrieval:
            assert self.retriever is not None
            hits = self.retriever.retrieve(question, k=self.config.top_k)
            docs = [h.document for h in hits]
            template = PROMPT_VARIANTS[self.config.prompt_version]
            prompt = template.format(context=self._format_context(docs), question=question)
        else:
            docs = []
            prompt = DIRECT_PROMPT.format(question=question)

        answer = self._backend.chat(
            [{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return RagResult(
            question=question,
            answer=answer,
            contexts=[d.text for d in docs],
            retrieved=docs,
        )
