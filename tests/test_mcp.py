"""MCP connector surface."""

from __future__ import annotations

import pytest

from hybrid_rag.mcp import RetrieverConnector


def test_list_tools():
    names = {t["name"] for t in RetrieverConnector().list_tools()}
    assert names == {"hybrid_search", "multi_hop"}


def test_call_hybrid_search():
    out = RetrieverConnector().call_tool("hybrid_search", {"query": "how does BM25 rank", "k": 3})
    assert len(out["results"]) == 3
    assert "id" in out["results"][0]


def test_call_multi_hop():
    out = RetrieverConnector().call_tool("multi_hop", {"concept": "hybrid-search", "hops": 1})
    assert "hybrid-search" in out["documents"]


def test_unknown_tool_raises():
    with pytest.raises(ValueError):
        RetrieverConnector().call_tool("nope", {})
