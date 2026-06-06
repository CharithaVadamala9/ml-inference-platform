"""Tests for the RAG system under test.

These avoid any LLM/network call: dataset loading is pure, and retrieval uses
the local embedding model (downloaded once, then cached).
"""

import mlip.rag.pipeline as pipeline_mod
from mlip.rag.corpus import load_corpus, load_eval
from mlip.rag.pipeline import PROMPT_VARIANTS, RagConfig, RagPipeline
from mlip.rag.retriever import Retriever


class _EchoBackend:
    """A fake backend that echoes the prompt back, so we can inspect it."""

    name = "echo"
    model = "echo"

    def chat(self, messages, *, temperature=0.0, max_tokens=512):
        return messages[0]["content"]


def test_corpus_and_eval_load() -> None:
    corpus = load_corpus()
    evalset = load_eval()
    assert len(corpus) >= 10
    assert len(evalset) >= 10
    # Every eval example points at real corpus documents.
    corpus_ids = {d.id for d in corpus}
    for ex in evalset:
        assert ex.question and ex.ground_truth
        assert set(ex.source_ids) <= corpus_ids


def test_retriever_finds_relevant_document() -> None:
    hits = Retriever().retrieve("Why do we use dropout in neural networks?", k=3)
    assert hits[0].document.id == "dropout"
    # Scores are sorted descending.
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))


def test_prompt_variants_format() -> None:
    cfg = RagConfig()
    assert cfg.prompt_version in PROMPT_VARIANTS
    rendered = PROMPT_VARIANTS[cfg.prompt_version].format(context="CTX", question="Q?")
    assert "CTX" in rendered and "Q?" in rendered


def test_norag_pipeline_skips_retrieval(monkeypatch) -> None:
    # No Ollama: swap in an echo backend so we can inspect the prompt sent.
    monkeypatch.setattr(pipeline_mod, "get_backend", lambda *a, **k: _EchoBackend())
    pipe = RagPipeline(RagConfig(name="bare", use_retrieval=False))
    assert pipe.retriever is None  # no corpus/embeddings loaded in no-RAG mode
    result = pipe.answer("What is dropout?")
    assert result.contexts == []  # nothing retrieved
    # The direct prompt has no "Context:" block (that's only in RAG prompts).
    assert "Context:" not in result.answer
    assert "What is dropout?" in result.answer
