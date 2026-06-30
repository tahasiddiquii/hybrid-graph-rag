"""Ranked-retrieval metrics."""

from __future__ import annotations

from hybrid_rag.benchmark.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"b", "x"}, 3) == 0.5
    assert recall_at_k(["a", "b"], {"z"}, 2) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a"], {"z"}) == 0.0


def test_ndcg_perfect_and_partial():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == 1.0
    partial = ndcg_at_k(["x", "a"], {"a"}, 2)  # relevant only at rank 2
    assert 0.60 < partial < 0.64
