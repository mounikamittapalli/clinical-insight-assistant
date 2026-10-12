import numpy as np
import pytest

from cia.embeddings import HashingEmbedder
from cia.schema import Chunk
from cia.vectorstore import NumpyStore

TEXTS = [
    "metformin for type 2 diabetes",
    "albuterol inhaler for asthma",
    "lisinopril for hypertension",
]


def _chunks():
    return [Chunk(f"c{i}", f"d{i}", t, f"src:{i}") for i, t in enumerate(TEXTS)]


def _check(store_cls, tmp_path):
    emb = HashingEmbedder(128)
    store = store_cls(emb.dim)
    store.add(_chunks(), emb.embed(TEXTS))
    hit = store.search(emb.embed_query("asthma inhaler"), k=1)[0]
    assert hit.chunk.chunk_id == "c1"
    store.save(tmp_path / "idx")
    reloaded = store_cls.load(tmp_path / "idx")
    again = reloaded.search(emb.embed_query("asthma inhaler"), k=1)[0]
    assert again.chunk.chunk_id == "c1"
    assert np.isclose(again.score, hit.score)


def test_numpy_store_roundtrip(tmp_path):
    _check(NumpyStore, tmp_path)


def test_faiss_store_roundtrip(tmp_path):
    pytest.importorskip("faiss")
    from cia.vectorstore import FaissStore

    _check(FaissStore, tmp_path)
