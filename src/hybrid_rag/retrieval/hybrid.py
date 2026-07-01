"""Hybrid retriever: BM25 + dense, fused with RRF, with optional reranking."""

from __future__ import annotations

from hybrid_rag.retrieval.base import Document, ScoredDoc
from hybrid_rag.retrieval.bm25 import BM25Retriever
from hybrid_rag.retrieval.dense import DenseRetriever
from hybrid_rag.retrieval.fusion import lexical_rerank, reciprocal_rank_fusion


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self,
        documents: list[Document],
        *,
        candidate_k: int = 20,
        rerank: bool = False,
        weights: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self.documents = documents
        self.candidate_k = candidate_k
        self.rerank = rerank
        self._weights = list(weights)
        self.bm25 = BM25Retriever(documents)
        self.dense = DenseRetriever(documents)

    def search(self, query: str, k: int = 10) -> list[ScoredDoc]:
        lexical = self.bm25.search(query, self.candidate_k)
        semantic = self.dense.search(query, self.candidate_k)
        fused = reciprocal_rank_fusion([lexical, semantic], k=self.candidate_k, weights=self._weights)
        if self.rerank:
            return lexical_rerank(query, fused, k=k)
        return fused[:k]
