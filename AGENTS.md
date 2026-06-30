# Agent / contributor guide

Orientation for an AI agent or new contributor working in this repo.

## What this is

Hybrid (BM25 + dense + RRF) and graph retrieval with a labeled benchmark and an
MCP connector. **Fully offline by default — no API keys, no network, no model
downloads.** Dense retrieval uses a deterministic feature-hashing embedding so the
benchmark is reproducible bit-for-bit.

## Layout

```
src/hybrid_rag/
  text.py            tokenization + char-n-grams
  retrieval/         bm25 · dense · embeddings · fusion · hybrid · base
  graph/             property_graph (BFS, multi-hop, paths)
  benchmark/         metrics · runner (gate) · report
  mcp/               connector (MCP tool surface)
  cli.py             search / benchmark / graph / mcp / demo
data/                corpus.jsonl · qrels.jsonl · triples.jsonl
tests/               one file per module
reports/             benchmark_report_example.md (committed proof)
```

## Conventions

- Python 3.12. `from __future__ import annotations` at the top of every module.
- Lint/format with ruff (`make fmt`, `make lint`). Line length 110.
- Pure functions where possible; retrievers share a `.name` / `.search(query, k)`
  interface so they are interchangeable in the benchmark and fusion.
- **Never hardcode metrics.** Every number in the README/report is produced by
  `run_benchmark`. If you change retrieval, regenerate the report and update the
  README to match — do not hand-edit numbers.

## Definition of done

```bash
make lint      # ruff clean
make test      # all tests pass
make benchmark # gate exits 0
```

The same three checks run in CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

## Extending

- Real embeddings: implement the `dense` extra (`sentence-transformers`) behind the
  `DenseRetriever` interface.
- Real graph: implement the `graph` extra (`neo4j`) behind the `PropertyGraph` interface.
- New metric: add to `benchmark/metrics.py` + a test, then wire into `runner.py`.
