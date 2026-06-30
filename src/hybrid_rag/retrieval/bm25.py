"""BM25 lexical retriever (exact-term matching)."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from hybrid_rag.retrieval.base import Document, ScoredDoc, top_k
from hybrid_rag.text import content_tokens


class BM25Retriever:
    name = "bm25"

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self._ids = [d.id for d in documents]
        self._texts = {d.id: d.text for d in documents}
        self._bm25 = BM25Okapi([content_tokens(d.text) for d in documents])

    def search(self, query: str, k: int = 10) -> list[ScoredDoc]:
        scores = self._bm25.get_scores(content_tokens(query))
        scored = [
            ScoredDoc(doc_id, float(score), self._texts[doc_id])
            for doc_id, score in zip(self._ids, scores, strict=True)
        ]
        return top_k(scored, k)
