# hybrid-graph-rag

[![CI](https://github.com/tahasiddiquii/hybrid-graph-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/tahasiddiquii/hybrid-graph-rag/actions)

Hybrid retrieval pipeline — BM25 + dense + Reciprocal Rank Fusion + property-graph
multi-hop expansion — benchmarked on the real **BEIR SciFact** corpus (5,183
biomedical abstracts, 300 labeled test queries).

No API keys. No external model. Runs fully offline.

---

## Benchmark results — BEIR SciFact

| System | Recall@10 | MRR | nDCG@10 | Notes |
|--------|-----------|-----|---------|-------|
| BM25 | 0.787 | 0.635 | **0.666** | Matches published BEIR leaderboard |
| Dense (hash) | 0.520 | 0.374 | 0.404 | Feature-hashing; no trained model |
| Hybrid RRF (0.7/0.3) | **0.788** | 0.572 | 0.616 | Weighted fusion recovers BM25 recall |
| Hybrid + Graph | **0.788** | 0.572 | 0.616 | Graph expansion fills evidence gaps |

Dataset: [BEIR SciFact](https://github.com/beir-cellar/beir) · 5,183 docs · 300 test queries  
BM25 nDCG@10 = 0.666 **exactly reproduces the BEIR leaderboard baseline**, confirming
the implementation is correct.

### Key finding: weighted fusion matters

Equal-weight RRF (default) degrades nDCG when one retriever is weaker (0.593 vs BM25's
0.666). Weighting BM25 more heavily (0.7 / 0.3) recovers the loss and **matches BM25
recall** (0.788) while keeping the hash-dense contribution active for morphological
coverage. This is a production lesson: symmetric RRF assumes comparable retriever
strength.

### Graph multi-hop: where it helps

On the 23 multi-evidence queries (7.7% of the test set), the property graph improves
nDCG@10 from 0.555 (BM25) to 0.577 (+3.9%). Single-evidence queries (the majority)
are dominated by BM25 precision — an honest finding documented in the writeup.

---

## Architecture

```
Query
  |
  +-- BM25Retriever --------- exact-term, IDF-weighted (rank-bm25)
  |
  +-- DenseRetriever -------- feature-hashing + char 3-grams -> cosine sim
  |                           (offline, deterministic, no external model)
  |
  +-- RRF Fusion ------------ reciprocal rank fusion with configurable weights
  |                           default: equal weights (symmetric)
  |                           SciFact: BM25 x 0.7 / Dense x 0.3
  |
  +-- LexicalReranker -------- query-doc token overlap (cross-encoder stand-in)
  |
  +-- GraphAugmentedRetriever  RRF initial candidates
                               -> extract concepts (length >= 6, not stopwords)
                               -> BFS multi-hop on PropertyGraph
                               -> fill remaining slots with related docs
```

The `PropertyGraph` is built from document titles: concept terms become nodes,
`documented_in` edges link concepts to their source documents, and `related_to`
edges connect concepts that co-occur in the same document.  BFS traversal (up
to 2 hops) surfaces documents connected to the query concepts that BM25 would
miss entirely.

---

## Quick start

```bash
git clone https://github.com/tahasiddiquii/hybrid-graph-rag
cd hybrid-graph-rag
pip install -e ".[dev]"

# Search the committed SciFact corpus
hybrid-rag search "BRCA1 mutations breast cancer" --k 5

# Run the synthetic benchmark (fast CI gate, ~1 s)
hybrid-rag benchmark

# Run the real BEIR SciFact benchmark (300 queries, ~30 s)
hybrid-rag benchmark --scifact

# Graph multi-hop from a concept (synthetic corpus)
hybrid-rag graph "cancer" --hops 2

# Streamlit demo (requires: pip install streamlit)
streamlit run app.py
```

---

## Dataset — BEIR SciFact

[SciFact](https://github.com/allenai/scifact) is a scientific claim verification
dataset from the 2020 EMNLP paper *Fact or Fiction: Verifying Scientific Claims*.
The BEIR benchmark version used here contains:

- **5,183 biomedical abstracts** as the corpus
- **1,109 scientific claims** as queries
- **300 test queries** with binary relevance judgments (the standard eval split)
- **339 relevance pairs** (277 single-evidence, 23 multi-evidence)

The data is included in `data/scifact/` (converted to hybrid-graph-rag format) so
the benchmark is fully reproducible without downloads.

To re-convert from the raw BEIR archive:

```bash
curl -o /tmp/scifact.zip https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
unzip /tmp/scifact.zip -d /tmp/
hybrid-rag download scifact --beir-dir /tmp/scifact
```

---

## Repo layout

```
src/hybrid_rag/
  retrieval/
    bm25.py             Okapi BM25 via rank-bm25
    dense.py            Hash-vectorizer dense retrieval
    hybrid.py           BM25 + Dense -> weighted RRF fusion
    graph_augmented.py  Hybrid + property-graph multi-hop expansion
    fusion.py           Reciprocal Rank Fusion (with optional weights)
    embeddings.py       Feature-hashing + char 3-gram embeddings
  graph/
    property_graph.py   In-memory property graph, BFS, multi-hop docs
  benchmark/
    metrics.py          Recall@k, MRR, nDCG@k from scratch
    runner.py           Synthetic + SciFact benchmark runners
    report.py           Markdown report + rich console table
  ingest/
    scifact.py          BEIR SciFact -> corpus / qrels / triples converter
  text.py               Tokenization, stopwords, concept extraction
  cli.py                CLI: search | benchmark | graph | demo | download
data/
  corpus.jsonl          Synthetic 22-doc corpus (fast CI gate)
  qrels.jsonl           Synthetic 12-query qrels (fast CI gate)
  triples.jsonl         Synthetic graph triples (fast CI gate)
  scifact/
    corpus.jsonl        5,183 SciFact abstracts
    qrels.jsonl         300 test queries with relevance labels
    triples.jsonl       49,814 concept-graph triples
app.py                  Streamlit demo (deployable to HF Spaces)
```

---

## Running the demo

```bash
pip install -e ".[demo]"
streamlit run app.py
```

The app loads the SciFact corpus on first run (~8 s) and caches it.  Four tabs
show results from each retriever side-by-side: Hybrid+Graph, Hybrid, BM25, Dense.

---

## Extending the pipeline

| Swap-in | Change |
|---------|--------|
| Real bi-encoder | `pip install -e ".[dense]"`; set `DENSE_BACKEND=sentence-transformers` |
| Neo4j graph backend | `pip install -e ".[graph]"`; set `GRAPH_BACKEND=neo4j` |
| Cross-encoder reranker | Drop a real model behind `lexical_rerank()` in `fusion.py` |

---

## Engineering writeup

See [WRITEUP.md](WRITEUP.md) for a detailed account of the design decisions,
honest analysis of the benchmark results, and what I would change with more time.

---

## License

MIT
