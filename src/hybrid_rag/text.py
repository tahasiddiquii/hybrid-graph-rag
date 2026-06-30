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
