"""Ingest module — converts raw benchmark datasets to hybrid-graph-rag format."""

from __future__ import annotations

from hybrid_rag.ingest.scifact import convert_scifact

__all__ = ["convert_scifact"]
