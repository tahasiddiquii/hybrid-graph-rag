"""In-memory property graph: neighbours, BFS, multi-hop docs, paths."""

from __future__ import annotations

from hybrid_rag.graph import build_graph


def test_neighbors_and_bfs_distances():
    g = build_graph()
    combined = {e.obj for e in g.neighbors("hybrid-search", "combines")}
    assert combined == {"bm25", "dense-retrieval"}
    reach = dict(g.bfs("hybrid-search", hops=2))
    assert reach["bm25"] == 1
    assert reach["embeddings"] == 2


def test_multi_hop_docs_includes_neighbours():
    g = build_graph()
    docs = g.multi_hop_docs("hybrid-search", hops=1)
    assert "hybrid-search" in docs
    assert "bm25" in docs
    assert "dense-retrieval" in docs


def test_path_between_concepts():
    g = build_graph()
    path = g.path("hybrid-search", "embeddings")
    assert path is not None
    assert path[-1].obj == "embeddings"


def test_documented_in_is_not_traversed():
    g = build_graph()
    # 'cosine-similarity' has only a documented_in edge, which BFS must skip
    assert g.bfs("cosine-similarity", hops=2) == []
    # but its supporting document is still retrievable directly
    assert g.documents_for("cosine-similarity") == ["cosine-similarity"]
