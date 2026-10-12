import pytest

from cia.build_index import build_index
from cia.factory import build_retriever
from cia.retrieval import BM25Index, reciprocal_rank_fusion
from cia.schema import Chunk, RetrievedChunk


def _c(cid: str, text: str = "x") -> Chunk:
    return Chunk(cid, cid, text, f"src:{cid}")


def test_rrf_rewards_agreement():
    dense = [RetrievedChunk(_c("a"), 0.9), RetrievedChunk(_c("b"), 0.8)]
    sparse = [RetrievedChunk(_c("b"), 12.0), RetrievedChunk(_c("c"), 9.0)]
    fused = reciprocal_rank_fusion([dense, sparse])
    assert fused[0].chunk.chunk_id == "b"  # appears in both lists
    assert {f.chunk.chunk_id for f in fused} == {"a", "b", "c"}


def test_bm25_matches_exact_terms():
    chunks = [_c("1", "Metformin for diabetes"), _c("2", "Albuterol inhaler for asthma")]
    hits = BM25Index(chunks).search("albuterol", k=2)
    assert hits[0].chunk.chunk_id == "2"


@pytest.fixture
def retriever(cfg, embedder):
    build_index(cfg, embedder=embedder)
    return build_retriever(cfg, embedder=embedder)


def test_hybrid_retrieves_right_patient(retriever):
    hits = retriever.retrieve("What medications does patient p-002 take?", top_k=3)
    assert hits[0].chunk.source == "patient:p-002"


def test_retrieves_policy_document(retriever):
    hits = retriever.retrieve("continuous glucose monitor coverage insulin", top_k=2)
    assert hits[0].chunk.source.startswith("policy:")


def test_dense_only_mode(retriever):
    hits = retriever.retrieve("asthma", top_k=2, hybrid=False)
    assert len(hits) == 2


def test_build_index_stats(cfg, embedder):
    stats = build_index(cfg, embedder=embedder)
    assert stats["patients"] == 3 and stats["policy_documents"] == 1
    assert stats["chunks"] >= 4
