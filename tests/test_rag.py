"""Tests for the RAG system under test.

These avoid any LLM/network call: dataset loading is pure, and retrieval uses
the local embedding model (downloaded once, then cached).
"""

from mlip.rag.corpus import load_corpus, load_eval
from mlip.rag.pipeline import PROMPT_VARIANTS, RagConfig
from mlip.rag.retriever import Retriever


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
