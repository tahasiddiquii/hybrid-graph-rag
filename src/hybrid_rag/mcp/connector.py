"""MCP connector: expose hybrid + graph retrieval as Model Context Protocol tools.

This is a transport-agnostic stub: ``list_tools()`` returns MCP-style tool
definitions and ``call_tool()`` dispatches them. Wire these into the official
``mcp`` server SDK (stdio/SSE) to let any MCP client — Claude Desktop, an IDE
agent — retrieve over this corpus. Kept dependency-free so it runs and tests offline.
"""

from __future__ import annotations

from typing import Any

from hybrid_rag.graph import PropertyGraph, build_graph
from hybrid_rag.retrieval import HybridRetriever, load_corpus

_TOOLS = [
    {
        "name": "hybrid_search",
        "description": "Hybrid (BM25 + dense, RRF-fused) search over the corpus. Returns ranked document ids.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language query."},
                "k": {"type": "integer", "description": "Number of results.", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "multi_hop",
        "description": "Graph multi-hop lookup: documents supporting a concept and its neighbours.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "concept": {"type": "string", "description": "Start concept/entity id."},
                "hops": {"type": "integer", "description": "Traversal depth.", "default": 2},
            },
            "required": ["concept"],
        },
    },
]


class RetrieverConnector:
    """Bind the retriever + graph to the MCP tool surface."""

    def __init__(self, retriever: HybridRetriever | None = None, graph: PropertyGraph | None = None) -> None:
        documents = load_corpus()
        self.retriever = retriever or HybridRetriever(documents)
        self.graph = graph or build_graph()

    def list_tools(self) -> list[dict[str, Any]]:
        return _TOOLS

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "hybrid_search":
            k = int(arguments.get("k", 5))
            hits = self.retriever.search(arguments["query"], k=k)
            return {"results": [{"id": h.id, "score": round(h.score, 4)} for h in hits]}
        if name == "multi_hop":
            hops = int(arguments.get("hops", 2))
            docs = self.graph.multi_hop_docs(arguments["concept"], hops=hops)
            return {"documents": docs}
        raise ValueError(f"unknown tool: {name}")
