"""Shared data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A source document before chunking (a patient summary or a policy file)."""

    doc_id: str
    text: str
    source: str  # human-readable source, e.g. "patient:abc123" or "policy:NCD-220.1.pdf"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable piece of a document."""

    chunk_id: str
    doc_id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class Answer:
    question: str
    text: str
    citations: list[RetrievedChunk]
    contexts: list[RetrievedChunk]
    mode: str
    blocked: bool = False
    reason: str | None = None
