"""Small text utilities shared by BM25, metrics and the hashing embedder."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the to was were "
    "what which who with does do did this these those their there".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    return [t for t in tokens if not drop_stopwords or t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 2]
