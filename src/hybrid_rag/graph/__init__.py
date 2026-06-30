"""Graph exports + triples loader."""

from __future__ import annotations

import json
from pathlib import Path

from hybrid_rag.graph.property_graph import Edge, PropertyGraph

_TRIPLES = Path(__file__).resolve().parents[3] / "data" / "triples.jsonl"

__all__ = ["Edge", "PropertyGraph", "build_graph", "load_triples"]


def load_triples(path: Path | None = None) -> list[dict]:
    path = path or _TRIPLES
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_graph(triples: list[dict] | None = None) -> PropertyGraph:
    triples = triples if triples is not None else load_triples()
    graph = PropertyGraph()
    for t in triples:
        graph.add_edge(t["s"], t["r"], t["o"])
    return graph
