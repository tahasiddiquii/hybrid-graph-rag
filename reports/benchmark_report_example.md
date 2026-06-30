> Committed example output of `hybrid-rag benchmark` on the deterministic offline
> retrievers (hashing embeddings + BM25). Reproduce it verbatim with zero API keys.
> Every number is measured from the actual retrievers — nothing is hardcoded.

# Retrieval benchmark

Queries: 12 · k = 5 · corpus = offline hashing embeddings + BM25

| System | recall@5 | MRR | nDCG@5 |
| --- | --- | --- | --- |
| bm25 **(best)** | 0.958 | 0.958 | 0.927 |
| dense | 0.917 | 1.000 | 0.936 |
| hybrid **(best)** | 0.958 | 1.000 | 0.958 |
| hybrid+rerank **(best)** | 0.958 | 1.000 | 0.958 |

## Gate

| Metric | Value | Threshold | Pass |
| --- | --- | --- | --- |
| hybrid_recall@5 | 0.958 | 0.80 | ✅ |
| hybrid_ndcg@5 | 0.958 | 0.60 | ✅ |

> Fusion matches the stronger single retriever on recall@5 (Δ +0.000) while strictly
> improving ranking quality: nDCG@5 rises **+0.022** over the best component and MRR
> reaches **1.000**. RRF works on ranks, not raw BM25/cosine scores, so a strong lexical
> hit and a strong semantic hit reinforce instead of fighting — robust across query types.

## How to read this

- **BM25** wins on raw recall — exact-term matching is hard to beat when the query
  shares vocabulary with the relevant document.
- **Dense** (char-n-gram hashing embeddings) gets a perfect MRR by surfacing the right
  document first on paraphrased / morphologically-varied queries where BM25's exact
  matching slips.
- **Hybrid (RRF)** keeps BM25's recall *and* dense's first-rank precision, landing the
  best nDCG of all four systems. That robustness across query types — not a single
  headline number — is the reason to fuse.
