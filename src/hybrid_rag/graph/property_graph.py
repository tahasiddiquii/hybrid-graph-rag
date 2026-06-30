"""A tiny in-memory property graph for multi-hop retrieval.

Nodes are entities/concepts; directed edges are ``(subject) -[relation]-> (object)``.
A special ``documented_in`` relation links a concept to the corpus document that
defines it, so a multi-hop traversal can return supporting documents — the thing a
vector search alone struggles with on "A relates to B relates to C" questions.

The interface mirrors a property-graph store; install the ``graph`` extra and point
``GRAPH_BACKEND=neo4j`` to back it with Neo4j without changing call sites.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Edge:
    subject: str
    relation: str
    obj: str


@dataclass
class PropertyGraph:
    _out: dict[str, list[Edge]] = field(default_factory=lambda: defaultdict(list))
    _nodes: set[str] = field(default_factory=set)

    def add_edge(self, subject: str, relation: str, obj: str) -> None:
        edge = Edge(subject, relation, obj)
        self._out[subject].append(edge)
        self._nodes.update((subject, obj))

    @property
    def nodes(self) -> set[str]:
        return set(self._nodes)

    def neighbors(self, node: str, relation: str | None = None) -> list[Edge]:
        edges = self._out.get(node, [])
        return [e for e in edges if relation is None or e.relation == relation]

    def documents_for(self, node: str) -> list[str]:
        return [e.obj for e in self.neighbors(node, "documented_in")]

    def bfs(self, start: str, hops: int = 2) -> list[tuple[str, int]]:
        """Breadth-first reachable nodes within ``hops``, as (node, distance) pairs."""
        seen: dict[str, int] = {start: 0}
        queue: deque[str] = deque([start])
        order: list[tuple[str, int]] = []
        while queue:
            node = queue.popleft()
            dist = seen[node]
            if dist >= hops:
                continue
            for edge in self._out.get(node, []):
                if edge.relation == "documented_in":
                    continue
                if edge.obj not in seen:
                    seen[edge.obj] = dist + 1
                    order.append((edge.obj, dist + 1))
                    queue.append(edge.obj)
        return order

    def multi_hop_docs(self, start: str, hops: int = 2) -> list[str]:
        """Documents supporting the start concept and everything within ``hops`` of it."""
        docs: list[str] = list(self.documents_for(start))
        for node, _ in self.bfs(start, hops):
            for doc in self.documents_for(node):
                if doc not in docs:
                    docs.append(doc)
        return docs

    def path(self, start: str, goal: str, max_hops: int = 4) -> list[Edge] | None:
        """Shortest relation path from start to goal (BFS), or None."""
        if start == goal:
            return []
        prev: dict[str, Edge] = {}
        seen = {start}
        queue: deque[str] = deque([start])
        while queue:
            node = queue.popleft()
            for edge in self._out.get(node, []):
                if edge.relation == "documented_in" or edge.obj in seen:
                    continue
                prev[edge.obj] = edge
                if edge.obj == goal:
                    return _reconstruct(prev, start, goal)
                if len(seen) <= max_hops * len(self._nodes):
                    seen.add(edge.obj)
                    queue.append(edge.obj)
        return None


def _reconstruct(prev: dict[str, Edge], start: str, goal: str) -> list[Edge]:
    chain: list[Edge] = []
    node = goal
    while node != start and node in prev:
        edge = prev[node]
        chain.append(edge)
        node = edge.subject
    return list(reversed(chain))
