"""Vector stores: FAISS (default) and Chroma, behind one small interface.

Both store L2-normalized vectors and score by cosine similarity (inner product).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import numpy as np

from cia.schema import Chunk, RetrievedChunk


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    def search(self, query_vec: np.ndarray, k: int) -> list[RetrievedChunk]: ...

    def save(self, path: str | Path) -> None: ...


def _save_chunks(chunks: list[Chunk], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c)) + "\n")


def load_chunks(path: str | Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]


class NumpyStore:
    """Exact cosine search in NumPy. No extra dependencies; fine up to ~100k chunks."""

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype="float32")
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.vectors = np.vstack([self.vectors, vectors.astype("float32")])
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        scores = self.vectors @ query_vec.astype("float32")
        top = np.argsort(-scores)[:k]
        return [RetrievedChunk(self.chunks[i], float(scores[i])) for i in top]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        _save_chunks(self.chunks, path / "chunks.jsonl")

    @classmethod
    def load(cls, path: str | Path) -> NumpyStore:
        path = Path(path)
        vectors = np.load(path / "vectors.npy")
        store = cls(vectors.shape[1])
        store.add(load_chunks(path / "chunks.jsonl"), vectors)
        return store


class FaissStore:
    """FAISS IndexFlatIP over normalized vectors (exact cosine search)."""

    def __init__(self, dim: int):
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.index.add(vectors.astype("float32"))
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        scores, ids = self.index.search(query_vec.reshape(1, -1).astype("float32"), k)
        return [
            RetrievedChunk(self.chunks[i], float(s))
            for s, i in zip(scores[0], ids[0], strict=False)
            if i != -1
        ]

    def save(self, path: str | Path) -> None:
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "faiss.index"))
        _save_chunks(self.chunks, path / "chunks.jsonl")

    @classmethod
    def load(cls, path: str | Path) -> FaissStore:
        import faiss

        path = Path(path)
        index = faiss.read_index(str(path / "faiss.index"))
        store = cls(index.d)
        store.index = index
        store.chunks = load_chunks(path / "chunks.jsonl")
        return store


class ChromaStore:
    """Persistent Chroma collection using cosine distance."""

    def __init__(self, path: str | Path, collection: str = "clinical_insight"):
        import chromadb

        self.path = Path(path)
        self.client = chromadb.PersistentClient(path=str(self.path / "chroma"))
        self.collection = self.client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], vectors: np.ndarray, batch: int = 4000) -> None:
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            self.collection.add(
                ids=[c.chunk_id for c in part],
                embeddings=vectors[i : i + batch].tolist(),
                documents=[c.text for c in part],
                metadatas=[
                    {"doc_id": c.doc_id, "source": c.source, **_flat(c.metadata)} for c in part
                ],
            )

    def search(self, query_vec: np.ndarray, k: int) -> list[RetrievedChunk]:
        res = self.collection.query(query_embeddings=[query_vec.tolist()], n_results=k)
        out = []
        for cid, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0],
            strict=False,
        ):
            chunk = Chunk(cid, meta.get("doc_id", ""), text, meta.get("source", ""), dict(meta))
            out.append(RetrievedChunk(chunk, 1.0 - float(dist)))
        return out

    def save(self, path: str | Path) -> None:  # Chroma persists automatically
        return None


def _flat(meta: dict) -> dict:
    """Chroma metadata values must be str, int, float or bool."""
    return {k: v for k, v in meta.items() if isinstance(v, str | int | float | bool)}


def build_store(backend: str, dim: int, path: str | Path):
    if backend == "faiss":
        return FaissStore(dim)
    if backend == "chroma":
        return ChromaStore(path)
    if backend == "numpy":
        return NumpyStore(dim)
    raise ValueError(f"Unknown vector store backend: {backend}")


def load_store(backend: str, path: str | Path):
    if backend == "faiss":
        return FaissStore.load(path)
    if backend == "chroma":
        return ChromaStore(path)
    if backend == "numpy":
        return NumpyStore.load(path)
    raise ValueError(f"Unknown vector store backend: {backend}")
