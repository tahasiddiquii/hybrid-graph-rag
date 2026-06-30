"""Shared retrieval types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class ScoredDoc:
    id: str
    score: float
    text: str = ""


def top_k(scored: list[ScoredDoc], k: int) -> list[ScoredDoc]:
    return sorted(scored, key=lambda d: d.score, reverse=True)[:k]
