"""Tests for the SciFact dataset, concept extraction, and graph-augmented retriever."""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_rag.graph.property_graph import PropertyGraph
from hybrid_rag.retrieval import load_corpus
from hybrid_rag.retrieval.base import Document
from hybrid_rag.retrieval.graph_augmented import GraphAugmentedRetriever
from hybrid_rag.text import extract_concepts

_SCIFACT_CORPUS = Path(__file__).parents[1] / "data" / "scifact" / "corpus.jsonl"
_SCIFACT_TRIPLES = Path(__file__).parents[1] / "data" / "scifact" / "triples.jsonl"
_SCIFACT_QRELS = Path(__file__).parents[1] / "data" / "scifact" / "qrels.jsonl"


# ── extract_concepts ──────────────────────────────────────────────────────────


def test_extract_concepts_keeps_long_content_words():
    concepts = extract_concepts("Microstructural development of cerebral white matter")
    assert "cerebral" in concepts or "microstructural" in concepts or "development" in concepts


def test_extract_concepts_filters_stopwords():
    concepts = extract_concepts("The study of cancer mutations in breast tissue")
    assert "study" not in concepts  # in CONCEPT_STOP
    assert "cancer" in concepts or "mutations" in concepts or "breast" in concepts


def test_extract_concepts_respects_max_terms():
    concepts = extract_concepts("alpha beta gamma delta epsilon zeta eta theta", max_terms=3)
    assert len(concepts) <= 3


def test_extract_concepts_short_words_excluded():
    concepts = extract_concepts("a cat sat on the mat")
    assert concepts == []  # all words below length-6 threshold


# ── GraphAugmentedRetriever ───────────────────────────────────────────────────


def _mini_corpus() -> list[Document]:
    return [
        Document(id="d0", text="Cancer mutation BRCA1 tumor suppressor gene expression"),
        Document(id="d1", text="BRCA2 protein breast cancer genetic mutation risk"),
        Document(id="d2", text="Diffusion tensor MRI cerebral white matter imaging"),
        Document(id="d3", text="Neural network deep learning classification model"),
        Document(id="d4", text="Myelodysplastic syndrome hematopoietic suppression"),
    ]


def _mini_graph() -> PropertyGraph:
    g = PropertyGraph()
    g.add_edge("cancer", "documented_in", "d0")
    g.add_edge("cancer", "documented_in", "d1")
    g.add_edge("cancer", "related_to", "breast")
    g.add_edge("breast", "documented_in", "d1")
    return g


def test_graph_augmented_returns_up_to_k_results():
    docs = _mini_corpus()
    graph = _mini_graph()
    ret = GraphAugmentedRetriever(docs, graph)
    hits = ret.search("cancer mutation BRCA", k=4)
    assert 1 <= len(hits) <= 4


def test_graph_augmented_finds_initial_candidates():
    docs = _mini_corpus()
    graph = _mini_graph()
    ret = GraphAugmentedRetriever(docs, graph)
    hits = ret.search("cancer", k=5)
    ids = [h.id for h in hits]
    # At least one of the cancer-related docs should surface
    assert "d0" in ids or "d1" in ids


def test_graph_augmented_expands_via_graph():
    """Graph expansion should surface d1 when starting from d0 via 'cancer' node."""
    docs = _mini_corpus()
    graph = _mini_graph()
    ret = GraphAugmentedRetriever(docs, graph, bm25_weight=0.9)
    hits = ret.search("cancer", k=5)
    ids = [h.id for h in hits]
    # Both cancer docs should appear (direct and graph-expanded)
    assert "d0" in ids and "d1" in ids


# ── SciFact data files ────────────────────────────────────────────────────────


@pytest.mark.skipif(not _SCIFACT_CORPUS.exists(), reason="SciFact data not committed")
def test_scifact_corpus_size():
    docs = load_corpus(_SCIFACT_CORPUS)
    assert len(docs) == 5183


@pytest.mark.skipif(not _SCIFACT_QRELS.exists(), reason="SciFact data not committed")
def test_scifact_qrels_size():
    import json

    qrels = [json.loads(line) for line in _SCIFACT_QRELS.read_text().splitlines() if line.strip()]
    assert len(qrels) == 300


@pytest.mark.skipif(not _SCIFACT_TRIPLES.exists(), reason="SciFact data not committed")
def test_scifact_triples_load():
    from hybrid_rag.graph import build_graph, load_triples

    triples = load_triples(_SCIFACT_TRIPLES)
    graph = build_graph(triples)
    assert len(graph.nodes) > 1000  # expect thousands of concept nodes
