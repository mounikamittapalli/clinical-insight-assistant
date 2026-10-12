"""Recursive character chunking with overlap.

Splits on paragraph breaks first, then lines, then sentences, then words, so chunks
end on natural boundaries wherever possible.
"""

from __future__ import annotations

import re

from cia.schema import Chunk, Document

SEPARATORS = ["\n\n", "\n", r"(?<=[.!?])\s+", " "]


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not seps:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    parts = re.split(sep, text) if sep.startswith("(") else text.split(sep)
    joiner = " " if sep.startswith("(") else sep
    pieces: list[str] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        candidate = f"{buf}{joiner}{part}" if buf else part
        if len(candidate) <= size:
            buf = candidate
            continue
        if buf:
            pieces.append(buf)
        if len(part) > size:
            pieces.extend(_split(part, size, rest))
            buf = ""
        else:
            buf = part
    if buf:
        pieces.append(buf)
    return pieces


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Split ``text`` into chunks of at most ``chunk_size`` chars, carrying ``overlap`` chars."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    pieces = _split(text.strip(), chunk_size - overlap, SEPARATORS)
    chunks: list[str] = []
    for i, piece in enumerate(pieces):
        if i == 0 or overlap == 0:
            chunks.append(piece.strip())
            continue
        tail = pieces[i - 1][-overlap:]
        # start the overlap on a word boundary
        space = tail.find(" ")
        tail = tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail
        chunks.append(f"{tail} {piece}".strip())
    return [c for c in chunks if c]


def chunk_documents(docs: list[Document], chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Chunk documents. If a document has a ``title`` in its metadata, every chunk after the
    first is prefixed with it, so a chunk from the middle of a patient summary still says
    which patient it belongs to (a "contextual chunk header")."""
    chunks: list[Chunk] = []
    for doc in docs:
        title = doc.metadata.get("title")
        for i, text in enumerate(chunk_text(doc.text, chunk_size, overlap)):
            if title and not text.startswith(title):
                text = f"{title} | {text}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{i}",
                    doc_id=doc.doc_id,
                    text=text,
                    source=doc.source,
                    metadata={**doc.metadata, "chunk_index": i},
                )
            )
    return chunks
