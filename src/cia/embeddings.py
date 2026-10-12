"""Embedding models. All return L2-normalized float32 vectors, so dot product = cosine."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype("float32")


class SentenceTransformerEmbedder:
    """Wraps a Hugging Face sentence-transformers model (default: BAAI/bge-small-en-v1.5)."""

    # BGE models retrieve better when queries carry this instruction prefix.
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.dim = self.model.get_sentence_embedding_dimension()
        self._use_prefix = "bge" in model_name.lower()

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
        return _normalize(np.asarray(vecs))

    def embed_query(self, text: str) -> np.ndarray:
        query = f"{self.QUERY_PREFIX}{text}" if self._use_prefix else text
        return self.embed([query])[0]


class HashingEmbedder:
    """Dependency-free bag-of-words embedder. Used in tests and as an offline fallback."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        import zlib

        from cia.text import tokenize

        vec = np.zeros(self.dim, dtype="float32")
        for tok in tokenize(text):
            # crc32 is stable across processes, unlike Python's salted hash()
            vec[zlib.crc32(tok.encode()) % self.dim] += 1.0
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
