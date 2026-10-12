"""Hybrid retrieval: dense vectors + BM25 keywords, fused with Reciprocal Rank Fusion,
then optionally re-ranked by a cross-encoder.

Why hybrid: dense embeddings catch paraphrases ("high blood pressure" ~ "hypertension"),
while BM25 catches exact terms embeddings blur (drug names, ICD/SNOMED codes, patient IDs).
"""

from __future__ import annotations

from typing import Protocol

from cia.embeddings import Embedder
from cia.schema import Chunk, RetrievedChunk
from cia.text import tokenize


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        corpus = [tokenize(c.text) or ["_empty_"] for c in chunks]
        self.token_sets = [set(doc) for doc in corpus]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        tokens = tokenize(query)
        if not tokens or not self.chunks:
            return []
        scores = self.bm25.get_scores(tokens)
        query_set = set(tokens)
        # Keep only chunks sharing a query term. (Okapi IDF can be 0 or negative for terms
        # in half the corpus, so a positive-score filter would drop real matches.)
        matches = [i for i in range(len(scores)) if self.token_sets[i] & query_set]
        top = sorted(matches, key=lambda i: -scores[i])[:k]
        return [RetrievedChunk(self.chunks[i], float(scores[i])) for i in top]


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    """Combine ranked lists: score(d) = sum over lists of 1 / (k + rank(d))."""
    fused: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            cid = item.chunk.chunk_id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
            by_id[cid] = item.chunk
    ordered = sorted(fused.items(), key=lambda kv: -kv[1])
    return [RetrievedChunk(by_id[cid], score) for cid, score in ordered]


class Reranker(Protocol):
    def rerank(self, query: str, items: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class CrossEncoderReranker:
    """Scores each (query, chunk) pair jointly; slower than embeddings but more precise."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        from cia.llm import pick_device

        self.model = CrossEncoder(model_name, device=pick_device())

    def rerank(self, query: str, items: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not items:
            return []
        scores = self.model.predict([(query, it.chunk.text) for it in items])
        ranked = sorted(zip(items, scores, strict=False), key=lambda p: -float(p[1]))
        return [RetrievedChunk(it.chunk, float(s)) for it, s in ranked]


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store,
        bm25: BM25Index | None = None,
        reranker: Reranker | None = None,
        top_k: int = 5,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25
        self.reranker = reranker
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        known = bm25.chunks if bm25 is not None else getattr(vector_store, "chunks", [])
        self.patient_ids = {
            str(c.metadata["patient_id"]).lower()
            for c in known
            if c.metadata.get("patient_id")
        }

    def mentioned_patients(self, query: str) -> set[str]:
        q = query.lower()
        return {pid for pid in self.patient_ids if pid in q}

    @staticmethod
    def _filter_to_patients(
        items: list[RetrievedChunk], patients: set[str]
    ) -> list[RetrievedChunk]:
        """Metadata filter: when a question names a patient, drop other patients' chunks
        (policy chunks are kept). Embeddings alone are bad at telling two UUIDs apart."""
        kept = [
            it for it in items
            if it.chunk.metadata.get("type") != "patient"
            or str(it.chunk.metadata.get("patient_id", "")).lower() in patients
        ]
        return kept or items

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        hybrid: bool | None = None,
        rerank: bool | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.top_k
        use_hybrid = (self.bm25 is not None) if hybrid is None else (hybrid and self.bm25)
        use_rerank = (self.reranker is not None) if rerank is None else (rerank and self.reranker)

        dense = self.vector_store.search(self.embedder.embed_query(query), self.candidate_k)
        if use_hybrid:
            sparse = self.bm25.search(query, self.candidate_k)
            candidates = reciprocal_rank_fusion([dense, sparse], k=self.rrf_k)
        else:
            candidates = dense

        patients = self.mentioned_patients(query)
        if patients:
            candidates = self._filter_to_patients(candidates, patients)
            if use_hybrid:
                # make sure the named patient's chunks are candidates even if dense search
                # and BM25 both ranked them low
                extra = [
                    RetrievedChunk(c, 0.0) for c in self.bm25.chunks
                    if str(c.metadata.get("patient_id", "")).lower() in patients
                    and c.chunk_id not in {x.chunk.chunk_id for x in candidates}
                ]
                candidates = candidates + extra

        candidates = candidates[: self.candidate_k]
        if use_rerank:
            candidates = self.reranker.rerank(query, candidates)
        return candidates[:top_k]
