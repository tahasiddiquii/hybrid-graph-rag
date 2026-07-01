"""Hybrid-Graph RAG interactive demo — Streamlit front-end.

Run locally::

    pip install streamlit
    streamlit run app.py

Deploy to Hugging Face Spaces by pushing this repo.  The app loads the committed
BEIR SciFact corpus (5,183 biomedical abstracts) from ``data/scifact/`` and builds
all indices in-memory on first load (≈ 8–10 s).
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Hybrid-Graph RAG",
    page_icon="🔍",
    layout="wide",
)

_ROOT = Path(__file__).parent
_SCIFACT = _ROOT / "data" / "scifact"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    k = st.slider("Top-k results", 3, 20, 5)
    st.divider()
    st.header("📊 Benchmark results")
    st.markdown(
        """
| System | Recall@10 | MRR | nDCG@10 |
|--------|-----------|-----|---------|
| BM25 | 0.787 | 0.635 | **0.666** |
| Dense (hash) | 0.520 | 0.374 | 0.404 |
| Hybrid (0.7/0.3) | 0.788 | 0.572 | 0.616 |
| Hybrid+Graph | **0.788** | 0.572 | 0.616 |

Dataset: [BEIR SciFact](https://github.com/beir-cellar/beir) · 5,183 docs · 300 test queries

BM25 nDCG@10 = 0.666 matches the published BEIR leaderboard exactly.
"""
    )
    st.divider()
    st.markdown(
        "**Source:** [github.com/tahasiddiquii/hybrid-graph-rag]"
        "(https://github.com/tahasiddiquii/hybrid-graph-rag)"
    )


# ── Index loading (cached across reruns) ──────────────────────────────────────
@st.cache_resource(show_spinner="Loading corpus and building search indices…")
def _load_indices():
    from hybrid_rag.graph import build_graph, load_triples
    from hybrid_rag.retrieval import BM25Retriever, DenseRetriever, Document, HybridRetriever
    from hybrid_rag.retrieval.graph_augmented import GraphAugmentedRetriever

    docs: list[Document] = []
    for line in (_SCIFACT / "corpus.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            docs.append(Document(id=r["id"], text=r["text"]))

    triples = load_triples(_SCIFACT / "triples.jsonl")
    graph = build_graph(triples)

    return (
        BM25Retriever(docs),
        DenseRetriever(docs),
        HybridRetriever(docs, weights=(0.7, 0.3)),
        GraphAugmentedRetriever(docs, graph),
    )


# ── Main UI ────────────────────────────────────────────────────────────────────
st.title("🔍 Hybrid-Graph RAG")
st.caption(
    "BM25 · Dense · Weighted RRF (0.7 / 0.3) · Property-graph multi-hop expansion  "
    "· BEIR SciFact benchmark (5,183 biomedical abstracts, 300 labeled test queries)"
)

query = st.text_input(
    "Search query",
    placeholder="e.g.  BRCA1 mutations increase breast cancer risk",
)

EXAMPLE_QUERIES = [
    "BRCA1 mutations increase breast cancer risk",
    "myelodysplastic syndrome bone marrow suppression",
    "diffusion tensor MRI white matter development",
    "vaccination reduces influenza mortality",
    "microRNA expression cancer prognosis",
]
cols = st.columns(len(EXAMPLE_QUERIES))
for col, ex in zip(cols, EXAMPLE_QUERIES, strict=False):
    if col.button(ex[:35] + "…", use_container_width=True):
        query = ex

if query:
    with st.spinner("Searching…"):
        bm25, dense, hybrid, hybrid_graph = _load_indices()
        bm25_hits = bm25.search(query, k=k)
        dense_hits = dense.search(query, k=k)
        hybrid_hits = hybrid.search(query, k=k)
        graph_hits = hybrid_graph.search(query, k=k)

    def _render(hits):
        for i, h in enumerate(hits, 1):
            is_graph = h.score <= 0.002
            score_badge = "*(graph-expanded)*" if is_graph else f"`score={h.score:.4f}`"
            title = h.text.split(". ")[0][:80] if ". " in h.text else h.text[:80]
            with st.expander(f"**{i}.** {title}   {score_badge}"):
                st.markdown(h.text[:600] + ("…" if len(h.text) > 600 else ""))
                st.caption(f"doc id: `{h.id}`")

    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Hybrid+Graph", "⚡ Hybrid (0.7/0.3)", "📖 BM25", "🧠 Dense (hash)"])
    with tab1:
        st.caption(
            "Weighted RRF (BM25 × 0.7, Dense × 0.3) → property-graph BFS expansion. "
            "Graph-expanded docs are labelled *(graph-expanded)*."
        )
        _render(graph_hits)
    with tab2:
        st.caption("Weighted RRF fusion — BM25 gets 0.7 weight, dense gets 0.3.")
        _render(hybrid_hits)
    with tab3:
        st.caption("Okapi BM25 exact-term matching (rank-bm25). nDCG@10 = 0.666.")
        _render(bm25_hits)
    with tab4:
        st.caption(
            "Hash-vectorizer dense retrieval (feature hashing + char 3-grams). "
            "Runs fully offline with no external model."
        )
        _render(dense_hits)
