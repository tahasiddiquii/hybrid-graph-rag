"""Reciprocal Rank Fusion (RRF) and a lightweight lexical reranker.

RRF combines several ranked lists without needing comparable scores: each list votes
for a document with weight ``1 / (k0 + rank)``. It is robust precisely because it uses
*ranks*, not raw BM25 / cosine magnitudes, so a strong lexical hit and a strong dense
hit reinforce each other.
"""

from __future__ import annotations

from hybrid_rag.retrieval.base import ScoredDoc, top_k
from hybrid_rag.text import content_tokens

RRF_K0 = 60


def reciprocal_rank_fusion(
    rankings: list[list[ScoredDoc]],
    k: int = 10,
    k0: int = RRF_K0,
) -> list[ScoredDoc]:
    """Fuse multiple ranked lists into one via RRF."""
    fused: dict[str, float] = {}
    texts: dict[str, str] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc.id] = fused.get(doc.id, 0.0) + 1.0 / (k0 + rank)
            texts.setdefault(doc.id, doc.text)
    merged = [ScoredDoc(doc_id, score, texts.get(doc_id, "")) for doc_id, score in fused.items()]
    return top_k(merged, k)


def lexical_rerank(query: str, candidates: list[ScoredDoc], k: int = 10) -> list[ScoredDoc]:
    """Re-score candidates by query/doc content-token overlap (a cheap cross-encoder stand-in).

    Install a real cross-encoder behind this signature for production reranking.
    """
    q = set(content_tokens(query))
    if not q:
        return top_k(candidates, k)
    rescored: list[ScoredDoc] = []
    for doc in candidates:
        d = set(content_tokens(doc.text))
        overlap = len(q & d) / len(q | d) if d else 0.0
        rescored.append(ScoredDoc(doc.id, overlap, doc.text))
    return top_k(rescored, k)
