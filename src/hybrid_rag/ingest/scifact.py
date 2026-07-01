"""Convert a local BEIR SciFact download to hybrid-graph-rag's data format.

Usage (CLI)::

    hybrid-rag download scifact --beir-dir /path/to/scifact

BEIR SciFact layout (after unzipping the official archive)::

    scifact/
      corpus.jsonl          # {"_id": "...", "title": "...", "text": "...", ...}
      queries.jsonl         # {"_id": "...", "text": "...", "metadata": {...}}
      qrels/
        test.tsv            # query-id \\t corpus-id \\t score

Output (written to ``out_dir``)::

    corpus.jsonl    {"id": "...", "text": "title. abstract"}
    qrels.jsonl     {"query": "...", "relevant": ["id", ...]}
    triples.jsonl   {"s": "concept", "r": "documented_in|related_to", "o": "..."}

Graph construction
------------------
We extract up to 4 salient content terms from each document title (length ≥ 6,
not in the concept stop-list) and:

* Add a ``documented_in`` edge from each concept node to its document(s).
* Add ``related_to`` edges between concept pairs that co-occur in the same
  document — but only when both concepts appear in at least 2 documents (i.e.
  we skip hapax-legomenon concept nodes that carry no cross-document signal).

This produces a concept-document bipartite graph extended with concept-concept
edges.  The graph is used by ``GraphAugmentedRetriever`` to expand a retrieval
result via BFS multi-hop traversal.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from hybrid_rag.text import extract_concepts

_DEFAULT_CAP = 8  # max docs per concept node (keeps graph focused)


def convert_scifact(
    beir_dir: str | Path,
    out_dir: str | Path,
    *,
    cap: int = _DEFAULT_CAP,
) -> dict[str, int]:
    """Convert BEIR SciFact files in *beir_dir* and write to *out_dir*.

    Returns a dict with counts: ``{"docs": N, "queries": N, "triples": N}``.
    """
    beir_dir = Path(beir_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── corpus ────────────────────────────────────────────────────────────────
    docs: dict[str, dict] = {}
    with open(beir_dir / "corpus.jsonl") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            title = row.get("title", "").strip()
            body = row.get("text", "").strip()
            text = f"{title}. {body}" if title else body
            docs[row["_id"]] = {"id": row["_id"], "text": text, "title": title}

    with open(out_dir / "corpus.jsonl", "w") as fh:
        for doc in docs.values():
            fh.write(json.dumps({"id": doc["id"], "text": doc["text"]}) + "\n")

    # ── queries ───────────────────────────────────────────────────────────────
    queries: dict[str, str] = {}
    with open(beir_dir / "queries.jsonl") as fh:
        for line in fh:
            if line.strip():
                q = json.loads(line)
                queries[q["_id"]] = q["text"]

    # ── qrels (test split only) ───────────────────────────────────────────────
    qrel_map: dict[str, list[str]] = defaultdict(list)
    with open(beir_dir / "qrels" / "test.tsv") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if int(row["score"]) > 0:
                qrel_map[row["query-id"]].append(row["corpus-id"])

    with open(out_dir / "qrels.jsonl", "w") as fh:
        for qid, rel_ids in qrel_map.items():
            if qid in queries:
                fh.write(json.dumps({"query": queries[qid], "relevant": rel_ids}) + "\n")

    # ── graph triples ─────────────────────────────────────────────────────────
    concept_to_docs: dict[str, list[str]] = defaultdict(list)
    for doc in docs.values():
        for c in extract_concepts(doc["title"]):
            concept_to_docs[c].append(doc["id"])

    triples: list[dict] = []
    added_pairs: set[tuple[str, str]] = set()

    for concept, doc_ids in concept_to_docs.items():
        if len(doc_ids) < 2:
            continue  # skip hapax concepts — no cross-document signal
        for doc_id in doc_ids[:cap]:
            triples.append({"s": concept, "r": "documented_in", "o": doc_id})

    for doc in docs.values():
        concepts = extract_concepts(doc["title"])
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i + 1 :]:
                pair = (min(c1, c2), max(c1, c2))
                if (
                    pair not in added_pairs
                    and len(concept_to_docs.get(c1, [])) >= 2
                    and len(concept_to_docs.get(c2, [])) >= 2
                ):
                    triples.append({"s": c1, "r": "related_to", "o": c2})
                    triples.append({"s": c2, "r": "related_to", "o": c1})
                    added_pairs.add(pair)

    with open(out_dir / "triples.jsonl", "w") as fh:
        for t in triples:
            fh.write(json.dumps(t) + "\n")

    return {"docs": len(docs), "queries": len(qrel_map), "triples": len(triples)}
