"""Hybrid + graph retrieval with a labeled benchmark and an MCP connector.

Public API:
    HybridRetriever  — BM25 + dense, RRF-fused, optional rerank.
    PropertyGraph    — in-memory multi-hop retrieval.
    run_benchmark    — recall@k / MRR / nDCG across retrievers + a CI gate.
    RetrieverConnector — MCP tool surface over the above.
"""

from __future__ import annotations

from hybrid_rag.benchmark import BenchmarkReport, run_benchmark
from hybrid_rag.graph import PropertyGraph, build_graph
from hybrid_rag.mcp import RetrieverConnector
from hybrid_rag.retrieval import Document, HybridRetriever, load_corpus

__all__ = [
    "BenchmarkReport",
    "Document",
    "HybridRetriever",
    "PropertyGraph",
    "RetrieverConnector",
    "build_graph",
    "load_corpus",
    "run_benchmark",
    "__version__",
]

__version__ = "0.1.0"
