"""Retrieval exports + corpus loader."""

from __future__ import annotations

import json
from pathlib import Path

from hybrid_rag.retrieval.base import Document, ScoredDoc, top_k
from hybrid_rag.retrieval.bm25 import BM25Retriever
from hybrid_rag.retrieval.dense import DenseRetriever
from hybrid_rag.retrieval.fusion import lexical_rerank, reciprocal_rank_fusion
from hybrid_rag.retrieval.hybrid import HybridRetriever

_CORPUS = Path(__file__).resolve().parents[3] / "data" / "corpus.jsonl"

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "Document",
    "HybridRetriever",
    "ScoredDoc",
    "lexical_rerank",
    "load_corpus",
    "reciprocal_rank_fusion",
    "top_k",
]


def load_corpus(path: Path | None = None) -> list[Document]:
    path = path or _CORPUS
    docs: list[Document] = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            docs.append(Document(id=row["id"], text=row["text"]))
    return docs
