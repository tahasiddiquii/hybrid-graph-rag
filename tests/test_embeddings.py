"""Deterministic hashing embeddings."""

from __future__ import annotations

import numpy as np

from hybrid_rag.retrieval.embeddings import embed


def test_deterministic_and_normalized():
    v1 = embed("hybrid search")
    v2 = embed("hybrid search")
    assert np.allclose(v1, v2)
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-5


def test_morphological_similarity_beats_unrelated():
    base = embed("ranking documents")
    near = embed("rank document")  # morphological variants
    far = embed("graph traversal cooking recipe")
    assert float(near @ base) > float(far @ base)
