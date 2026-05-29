"""Dense retriever: local sentence-transformer embeddings + cosine similarity.

The corpus here is tiny, so an exact in-memory cosine search over a normalized
embedding matrix is both simplest and fastest. The interface (`retrieve`) is
what would stay the same if this were swapped for FAISS or a vector DB.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from mlip.rag.corpus import Document, load_corpus

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class Retrieved:
    document: Document
    score: float


class Retriever:
    def __init__(
        self,
        documents: list[Document] | None = None,
        embed_model: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        self.documents = documents if documents is not None else load_corpus()
        self.embed_model = embed_model

    @cached_property
    def _model(self):
        # Imported lazily so importing this module is cheap (no torch load).
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.embed_model)

    @cached_property
    def _matrix(self) -> np.ndarray:
        texts = [d.text for d in self.documents]
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def retrieve(self, query: str, k: int = 3) -> list[Retrieved]:
        q = np.asarray(self._model.encode([query], normalize_embeddings=True), dtype=np.float32)[0]
        scores = self._matrix @ q  # cosine similarity (vectors are normalized)
        top = np.argsort(-scores)[:k]
        return [Retrieved(self.documents[i], float(scores[i])) for i in top]
