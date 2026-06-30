"""BM25, dense, and hybrid retrievers over the corpus."""

from __future__ import annotations

from hybrid_rag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    load_corpus,
)


def test_bm25_finds_exact_term():
    hits = BM25Retriever(load_corpus()).search("BM25 ranking function", k=3)
    assert "bm25" in [h.id for h in hits]


def test_dense_finds_paraphrase():
    hits = DenseRetriever(load_corpus()).search("matching paraphrases not exact keywords", k=3)
    assert "semantic-search" in [h.id for h in hits]


def test_hybrid_returns_k_and_relevant():
    hits = HybridRetriever(load_corpus()).search("fuse lexical and semantic retrievers", k=5)
    assert len(hits) == 5
    ids = [h.id for h in hits]
    assert "rrf" in ids or "hybrid-search" in ids


def test_hybrid_rerank_runs():
    hits = HybridRetriever(load_corpus(), rerank=True).search("approximate nearest neighbour search", k=3)
    assert len(hits) == 3
