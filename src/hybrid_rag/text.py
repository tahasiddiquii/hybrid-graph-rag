"""Tokenization helpers shared by the lexical and dense retrievers."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "by",
    "at",
    "as",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "from",
    "into",
    "how",
    "what",
    "which",
    "does",
    "do",
    "can",
    "you",
    "your",
    "we",
    "they",
    "i",
    "me",
    "my",
}


def word_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric word tokens."""
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Word tokens with stopwords removed (used for lexical scoring)."""
    return [t for t in word_tokens(text) if t not in _STOPWORDS]


def char_ngrams(token: str, n: int = 3) -> list[str]:
    """Padded character n-grams of a token, e.g. 'bm25' -> '#bm', 'bm2', 'm25', '25#'."""
    s = f"#{token}#"
    if len(s) < n:
        return [s]
    return [s[i : i + n] for i in range(len(s) - n + 1)]


# Extended stop-list for concept extraction (superset of _STOPWORDS).
_CONCEPT_STOP = _STOPWORDS | frozenset(
    [
        "study",
        "studies",
        "results",
        "result",
        "analysis",
        "effect",
        "effects",
        "treatment",
        "patients",
        "human",
        "compared",
        "control",
        "normal",
        "increased",
        "decreased",
        "associated",
        "significant",
        "significantly",
        "suggest",
        "indicates",
        "observed",
        "important",
        "potential",
        "novel",
        "recent",
        "previous",
        "current",
        "first",
        "second",
        "third",
        "known",
        "using",
        "used",
        "based",
        "single",
        "expression",
        "protein",
        "proteins",
        "function",
        "activity",
        "however",
        "addition",
        "despite",
        "although",
        "following",
        "major",
        "total",
        "different",
        "shows",
        "shown",
        "identify",
        "identified",
        "induced",
        "induces",
        "reveal",
        "reveals",
        "after",
        "before",
        "within",
        "between",
        "without",
        "during",
        "further",
        "toward",
        "towards",
        "might",
        "could",
        "would",
        "should",
        "receptor",
        "receptors",
        "tissue",
        "tissues",
        "response",
        "responses",
        "group",
        "groups",
        "level",
        "levels",
        "role",
        "roles",
        "assay",
        "assays",
        "showed",
        "show",
        "examine",
        "examined",
        "present",
        "presented",
        "involved",
        "involve",
    ]
)


def extract_concepts(text: str, max_terms: int = 4) -> list[str]:
    """Salient content terms (length ≥ 6, not common words) for graph lookup."""
    return [t for t in word_tokens(text) if len(t) >= 6 and t not in _CONCEPT_STOP][:max_terms]
