"""Dense retriever over the deterministic hashing embeddings."""

from __future__ import annotations

from hybrid_rag.retrieval.base import Document, ScoredDoc, top_k
from hybrid_rag.retrieval.embeddings import cosine, embed, embed_many


class DenseRetriever:
    name = "dense"

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self._ids = [d.id for d in documents]
        self._texts = [d.text for d in documents]
        self._matrix = embed_many(self._texts)

    def search(self, query: str, k: int = 10) -> list[ScoredDoc]:
        sims = cosine(embed(query), self._matrix)
        scored = [
            ScoredDoc(doc_id, float(sim), text)
            for doc_id, text, sim in zip(self._ids, self._texts, sims, strict=True)
        ]
        return top_k(scored, k)
