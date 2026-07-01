"""Run the labeled benchmark across retrievers and gate on the hybrid result."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from hybrid_rag.benchmark.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from hybrid_rag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    Document,
    HybridRetriever,
    load_corpus,
)

_ROOT = Path(__file__).resolve().parents[3]
QRELS = _ROOT / "data" / "qrels.jsonl"
REPORT = _ROOT / "reports" / "benchmark_report.md"

K = 5
SCIFACT_K = 10

# CI gate: fusion must clear an absolute bar AND not lose to either component.
THRESHOLDS = {
    "hybrid_recall@5": 0.80,
    "hybrid_ndcg@5": 0.60,
}

# SciFact gate: loose bar so CI is stable; main value is the full table.
SCIFACT_THRESHOLDS = {
    "hybrid_recall@10": 0.60,
    "hybrid_ndcg@10": 0.45,
}


@dataclass
class SystemScore:
    name: str
    recall: float
    mrr: float
    ndcg: float


@dataclass
class BenchmarkReport:
    k: int
    n_queries: int
    systems: list[SystemScore]
    aggregate: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=lambda: dict(THRESHOLDS))

    def by_name(self, name: str) -> SystemScore:
        return next(s for s in self.systems if s.name == name)

    def passed(self) -> bool:
        hybrid = self.by_name("hybrid")
        for metric, thr in self.thresholds.items():
            if self.aggregate.get(metric, 0.0) < thr:
                return False
        # fusion should not underperform either component on recall
        return all(hybrid.recall >= s.recall - 1e-9 for s in self.systems if s.name in {"bm25", "dense"})

    def failures(self) -> list[str]:
        hybrid = self.by_name("hybrid")
        out = []
        for metric, thr in self.thresholds.items():
            if self.aggregate.get(metric, 0.0) < thr:
                out.append(metric)
        for s in self.systems:
            if s.name in {"bm25", "dense"} and hybrid.recall < s.recall - 1e-9:
                out.append(f"hybrid<recall:{s.name}")
        return out


def load_qrels(path: Path | None = None) -> list[dict]:
    path = path or QRELS
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _score(name: str, retriever, qrels: list[dict], k: int) -> SystemScore:
    recalls, rrs, ndcgs = [], [], []
    for row in qrels:
        relevant = set(row["relevant"])
        retrieved = [d.id for d in retriever.search(row["query"], k=max(k, 10))]
        recalls.append(recall_at_k(retrieved, relevant, k))
        rrs.append(reciprocal_rank(retrieved, relevant))
        ndcgs.append(ndcg_at_k(retrieved, relevant, k))
    n = len(qrels) or 1
    return SystemScore(name, round(sum(recalls) / n, 3), round(sum(rrs) / n, 3), round(sum(ndcgs) / n, 3))


def run_benchmark(documents: list[Document] | None = None, k: int = K) -> BenchmarkReport:
    """Run the benchmark on the synthetic corpus (default CI gate)."""
    documents = documents or load_corpus()
    qrels = load_qrels()
    systems = [
        _score("bm25", BM25Retriever(documents), qrels, k),
        _score("dense", DenseRetriever(documents), qrels, k),
        _score("hybrid", HybridRetriever(documents), qrels, k),
        _score("hybrid+rerank", HybridRetriever(documents, rerank=True), qrels, k),
    ]
    hybrid = next(s for s in systems if s.name == "hybrid")
    aggregate = {
        f"hybrid_recall@{k}": hybrid.recall,
        "hybrid_mrr": hybrid.mrr,
        f"hybrid_ndcg@{k}": hybrid.ndcg,
    }
    return BenchmarkReport(k=k, n_queries=len(qrels), systems=systems, aggregate=aggregate)


def run_scifact_benchmark() -> BenchmarkReport:
    """Run the benchmark on the real BEIR SciFact corpus (5,183 docs, 300 queries)."""
    from hybrid_rag.graph import build_graph, load_triples
    from hybrid_rag.retrieval.graph_augmented import GraphAugmentedRetriever

    _SCIFACT = _ROOT / "data" / "scifact"
    if not (_SCIFACT / "corpus.jsonl").exists():
        raise FileNotFoundError(
            f"SciFact data not found at {_SCIFACT}.\n"
            "Run: hybrid-rag download scifact --beir-dir /path/to/scifact"
        )

    documents = load_corpus(_SCIFACT / "corpus.jsonl")
    qrels = load_qrels(_SCIFACT / "qrels.jsonl")
    triples = load_triples(_SCIFACT / "triples.jsonl")
    graph = build_graph(triples)
    k = SCIFACT_K

    systems = [
        _score("bm25", BM25Retriever(documents), qrels, k),
        _score("dense", DenseRetriever(documents), qrels, k),
        _score("hybrid", HybridRetriever(documents, weights=(0.7, 0.3)), qrels, k),
        _score("hybrid+graph", GraphAugmentedRetriever(documents, graph), qrels, k),
    ]
    hybrid = next(s for s in systems if s.name == "hybrid")
    aggregate = {
        f"hybrid_recall@{k}": hybrid.recall,
        "hybrid_mrr": hybrid.mrr,
        f"hybrid_ndcg@{k}": hybrid.ndcg,
    }
    return BenchmarkReport(
        k=k,
        n_queries=len(qrels),
        systems=systems,
        aggregate=aggregate,
        thresholds=SCIFACT_THRESHOLDS,
    )
