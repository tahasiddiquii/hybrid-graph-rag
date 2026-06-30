"""Command-line entry point: ``hybrid-rag search|benchmark|graph|mcp|demo``."""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_search(args: argparse.Namespace) -> int:
    from hybrid_rag.retrieval import HybridRetriever, load_corpus

    retriever = HybridRetriever(load_corpus(), rerank=args.rerank)
    for hit in retriever.search(args.query, k=args.k):
        print(f"{hit.score:7.4f}  {hit.id:18}  {hit.text[:70]}")
    return 0


def _cmd_benchmark(_: argparse.Namespace) -> int:
    from hybrid_rag.benchmark import print_table, run_benchmark, write_report

    report = run_benchmark()
    path = write_report(report)
    print_table(report)
    print(f"\nReport written to {path}")
    if not report.passed():
        print(f"BENCHMARK GATE FAILED: {report.failures()}")
        return 1
    print("BENCHMARK GATE PASSED")
    return 0


def _cmd_graph(args: argparse.Namespace) -> int:
    from hybrid_rag.graph import build_graph

    graph = build_graph()
    reachable = graph.bfs(args.concept, hops=args.hops)
    docs = graph.multi_hop_docs(args.concept, hops=args.hops)
    print(json.dumps({"concept": args.concept, "reachable": reachable, "documents": docs}, indent=2))
    return 0


def _cmd_mcp(_: argparse.Namespace) -> int:
    from hybrid_rag.mcp import RetrieverConnector

    connector = RetrieverConnector()
    print(json.dumps({"tools": connector.list_tools()}, indent=2))
    print("\nexample call:")
    print(
        json.dumps(connector.call_tool("hybrid_search", {"query": "how does BM25 rank?", "k": 3}), indent=2)
    )
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    from hybrid_rag.retrieval import HybridRetriever, load_corpus

    retriever = HybridRetriever(load_corpus())
    for q in ["how does ranking work", "detecting hallucinations", "fuse multiple retrievers"]:
        top = retriever.search(q, k=3)
        print(f"\nQ: {q}")
        for hit in top:
            print(f"   {hit.score:7.4f}  {hit.id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hybrid-rag", description="Hybrid + graph retrieval with a benchmark."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("search", help="Hybrid search a query.")
    s.add_argument("query")
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--rerank", action="store_true")
    s.set_defaults(func=_cmd_search)

    b = sub.add_parser("benchmark", help="Run the labeled retrieval benchmark + gate.")
    b.set_defaults(func=_cmd_benchmark)

    g = sub.add_parser("graph", help="Multi-hop graph lookup from a concept.")
    g.add_argument("concept")
    g.add_argument("--hops", type=int, default=2)
    g.set_defaults(func=_cmd_graph)

    m = sub.add_parser("mcp", help="Print the MCP tool definitions + an example call.")
    m.set_defaults(func=_cmd_mcp)

    d = sub.add_parser("demo", help="A few example searches.")
    d.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
