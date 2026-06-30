"""End-to-end retrieval benchmark + CI gate."""

from __future__ import annotations

from hybrid_rag.benchmark import run_benchmark


def test_benchmark_gate_passes():
    report = run_benchmark()
    assert report.passed(), f"gate failed on: {report.failures()}"


def test_hybrid_not_worse_than_components_on_recall():
    report = run_benchmark()
    hybrid = report.by_name("hybrid")
    assert hybrid.recall >= report.by_name("bm25").recall - 1e-9
    assert hybrid.recall >= report.by_name("dense").recall - 1e-9


def test_hybrid_has_best_ndcg():
    report = run_benchmark()
    hybrid = report.by_name("hybrid")
    best_component = max(report.by_name("bm25").ndcg, report.by_name("dense").ndcg)
    assert hybrid.ndcg >= best_component
