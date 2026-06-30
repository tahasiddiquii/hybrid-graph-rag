"""Deterministic, offline embeddings via the hashing trick.

Word tokens *and* character 3-grams are feature-hashed (with signed buckets) into a
fixed-dimension L2-normalized vector. Character n-grams give the dense space a
morphological signal — ``rank``/``ranking``/``ranked`` land near each other — which is
genuinely complementary to BM25's exact-term matching, so fusing the two helps.

This needs no model and no network, which is what makes the benchmark reproducible.
Install the ``dense`` extra and set ``DENSE_BACKEND=sentence-transformers`` to swap in
a real embedding model behind the same interface.
"""

from __future__ import annotations

import hashlib

import numpy as np

from hybrid_rag.text import char_ngrams, content_tokens

DIM = 256


def _hash(feature: str) -> int:
    return int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")


def _features(text: str) -> list[str]:
    tokens = content_tokens(text)
    feats: list[str] = list(tokens)
    for tok in tokens:
        feats.extend(f"c:{ng}" for ng in char_ngrams(tok))
    return feats


def embed(text: str, dim: int = DIM) -> np.ndarray:
    """Feature-hashed, L2-normalized embedding of a piece of text."""
    vec = np.zeros(dim, dtype=np.float32)
    for feat in _features(text):
        h = _hash(feat)
        idx = h % dim
        sign = 1.0 if (h >> 17) & 1 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0.0 else vec


def embed_many(texts: list[str], dim: int = DIM) -> np.ndarray:
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    return np.vstack([embed(t, dim) for t in texts])


def cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of a query vector against rows of an (already-normalized) matrix."""
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    return matrix @ query
