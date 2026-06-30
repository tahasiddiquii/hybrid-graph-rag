"""Reciprocal Rank Fusion and the lexical reranker."""

from __future__ import annotations

from hybrid_rag.retrieval.base import ScoredDoc
from hybrid_rag.retrieval.fusion import lexical_rerank, reciprocal_rank_fusion


def test_rrf_rewards_agreement_across_lists():
    list_a = [ScoredDoc("x", 9.0), ScoredDoc("y", 1.0)]
    list_b = [ScoredDoc("x", 0.8), ScoredDoc("z", 0.2)]
    fused = reciprocal_rank_fusion([list_a, list_b], k=3)
    assert fused[0].id == "x"  # ranks highly in both lists


def test_lexical_rerank_prefers_overlap():
    candidates = [
        ScoredDoc("a", 0.9, "graph traversal multi hop reasoning"),
        ScoredDoc("b", 0.1, "bm25 ranking by term frequency"),
    ]
    out = lexical_rerank("bm25 ranking", candidates, k=2)
    assert out[0].id == "b"
