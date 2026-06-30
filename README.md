# hybrid-graph-rag

> **Hybrid + graph retrieval, measured.** BM25 + dense retrieval fused with
> Reciprocal Rank Fusion (and optional reranking), an in-memory **property graph**
> for multi-hop retrieval, a labeled **benchmark** (recall@k · MRR · nDCG), and an
> **MCP connector** to expose it all as tools. Runs fully offline, **zero API keys**.

[![CI](https://github.com/tahasiddiquii/hybrid-graph-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/tahasiddiquii/hybrid-graph-rag/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Production RAG quality lives and dies on retrieval. This repo implements the two
techniques that move the needle most — **hybrid (lexical + dense) fusion** and
**graph multi-hop** — and, crucially, ships a benchmark so every claim is a number
you can reproduce, not a vibe.

## What this demonstrates

| Capability | Where |
| --- | --- |
| BM25 lexical retrieval | [src/hybrid_rag/retrieval/bm25.py](src/hybrid_rag/retrieval/bm25.py) |
| Dense retrieval (offline hashing embeddings, char-n-gram) | [src/hybrid_rag/retrieval/dense.py](src/hybrid_rag/retrieval/dense.py) |
| Reciprocal Rank Fusion + lexical reranker | [src/hybrid_rag/retrieval/fusion.py](src/hybrid_rag/retrieval/fusion.py) |
| Hybrid retriever orchestration | [src/hybrid_rag/retrieval/hybrid.py](src/hybrid_rag/retrieval/hybrid.py) |
| In-memory property graph + multi-hop | [src/hybrid_rag/graph/property_graph.py](src/hybrid_rag/graph/property_graph.py) |
| Benchmark: recall@k · MRR · nDCG + CI gate | [src/hybrid_rag/benchmark/](src/hybrid_rag/benchmark/) |
| MCP connector (tool schema + dispatch) | [src/hybrid_rag/mcp/connector.py](src/hybrid_rag/mcp/connector.py) |

## Architecture

```mermaid
flowchart TD
    Q[query] --> BM[BM25<br/>exact terms]
    Q --> DN[dense<br/>char-n-gram embeddings]
    BM --> RRF[Reciprocal Rank Fusion]
    DN --> RRF
    RRF --> RR[optional lexical rerank]
    RR --> R[ranked documents]

    subgraph graphretr[Graph retrieval]
        C[concept] --> G[(property graph)]
        G -->|multi-hop traverse| D[supporting documents]
    end

    subgraph eval[Benchmark]
        QR[qrels.jsonl] --> BENCH[run_benchmark]
        BENCH --> REP[recall@k · MRR · nDCG + gate]
    end

    R -.evaluated by.-> BENCH
    R --> MCP[[MCP tools:<br/>hybrid_search · multi_hop]]
    D --> MCP
```

## Quickstart

```bash
make dev                          # venv + install -e ".[dev]"  (Python 3.12)

hybrid-rag search "fuse lexical and semantic retrievers"
hybrid-rag benchmark              # recall@k / MRR / nDCG + CI gate
hybrid-rag graph hybrid-search --hops 2   # multi-hop document gathering
hybrid-rag mcp                    # print the MCP tool definitions + an example call
```

Everything is deterministic and offline — no model downloads, no keys, no network.

## The benchmark

`hybrid-rag benchmark` scores every retriever on a labeled query set
([full report](reports/benchmark_report_example.md)):

| System | recall@5 | MRR | nDCG@5 |
| --- | --- | --- | --- |
| bm25 | 0.958 | 0.958 | 0.927 |
| dense | 0.917 | 1.000 | 0.936 |
| **hybrid** | **0.958** | **1.000** | **0.958** |
| hybrid+rerank | 0.958 | 1.000 | 0.958 |

**Hybrid keeps BM25's recall and dense's first-rank precision, landing the best
nDCG of all four systems** (`+0.022` over the best single retriever). RRF operates on
*ranks*, not raw BM25/cosine scores, so a strong lexical hit and a strong semantic hit
reinforce instead of fighting — robust across query types. Numbers are measured, not
asserted; re-run to reproduce them exactly.

## Graph multi-hop

The property graph answers "A relates to B relates to C" questions that single-vector
search struggles with. From a concept it traverses relationships and returns the
supporting documents:

```bash
$ hybrid-rag graph hybrid-search --hops 2
{
  "concept": "hybrid-search",
  "reachable": [["bm25", 1], ["dense-retrieval", 1], ["reranking", 1],
                ["inverted-index", 2], ["tokenization", 2], ["embeddings", 2]],
  "documents": ["hybrid-search", "bm25", "dense-retrieval", "reranking",
                "inverted-index", "tokenization", "embeddings"]
}
```

## MCP connector

`RetrieverConnector` exposes `hybrid_search` and `multi_hop` as Model Context
Protocol tools (schema + dispatch), ready to register with the official `mcp` server
SDK so any MCP client — an IDE agent, Claude Desktop — can retrieve over this corpus.
Kept dependency-free so it runs and tests offline.

## Design decisions

- **Offline dense retrieval via the hashing trick.** Word tokens *and* character
  3-grams are feature-hashed into an L2-normalized vector. The char-n-grams give a
  morphological signal (`rank`≈`ranking`≈`ranked`) that genuinely complements BM25's
  exact matching — which is why fusion helps. Swap in `sentence-transformers` (the
  `dense` extra) behind the same interface for real semantics.
- **RRF over score-mixing.** Rank-based fusion needs no score normalization between
  incomparable retrievers and is hard to destabilize.
- **Honest metrics.** Every number is measured by `run_benchmark` from the actual
  retrievers; nothing is hardcoded. Hybrid *ties* BM25 on recall here — the win is in
  ranking quality (nDCG/MRR), and the report says so plainly.
- **Pluggable backends.** Dense → sentence-transformers, graph → Neo4j (the `graph`
  extra), all behind the in-repo interfaces.

## Layout

```
src/hybrid_rag/
  retrieval/   bm25 · dense · embeddings · fusion · hybrid
  graph/       property_graph (BFS, multi-hop, paths)
  benchmark/   metrics · runner · report
  mcp/         connector (MCP tool surface)
  cli.py
data/          corpus.jsonl · qrels.jsonl · triples.jsonl
reports/       benchmark_report_example.md
```

## Related repositories

Part of a series on production LLM engineering:

- [ai-harness](https://github.com/tahasiddiquii/ai-harness) — multi-stage agent harness (routing, guardrails, tools, evals).
- [llm-eval-observability](https://github.com/tahasiddiquii/llm-eval-observability) — RAG evaluation + Langfuse observability.
- [llm-guardrails-redteam](https://github.com/tahasiddiquii/llm-guardrails-redteam) — guardrails + red-team harness.
- **hybrid-graph-rag** — this repo.

## License

MIT © 2026 Taha Siddiqui
