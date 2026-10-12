"""Assemble a ready-to-use RAG pipeline from config (used by the API and evaluation)."""

from __future__ import annotations

from pathlib import Path

from cia.config import Config
from cia.rag import RAGPipeline
from cia.retrieval import BM25Index, CrossEncoderReranker, HybridRetriever
from cia.vectorstore import load_chunks, load_store


def build_retriever(cfg: Config, embedder=None, reranker=None) -> HybridRetriever:
    index_dir = Path(cfg.paths.index_dir)
    if not (index_dir / "chunks.jsonl").exists():
        raise FileNotFoundError(f"No index in {index_dir}. Run `python -m cia.build_index` first.")

    if embedder is None:
        from cia.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(cfg.embeddings.model, cfg.embeddings.batch_size)

    store = load_store(cfg.vector_store.backend, index_dir)
    r = cfg.retrieval
    bm25 = BM25Index(load_chunks(index_dir / "chunks.jsonl")) if r.hybrid else None
    if reranker is None and r.rerank:
        reranker = CrossEncoderReranker(r.reranker_model)

    return HybridRetriever(
        embedder,
        store,
        bm25=bm25,
        reranker=reranker,
        top_k=r.top_k,
        candidate_k=r.candidate_k,
        rrf_k=r.rrf_k,
    )


def build_generator(cfg: Config, use_adapter: bool | None = None):
    from cia.llm import HFGenerator

    use_adapter = cfg.llm.load_adapter if use_adapter is None else use_adapter
    adapter = cfg.paths.adapter_dir if use_adapter else None
    return HFGenerator(
        cfg.llm.model,
        adapter_path=adapter,
        max_new_tokens=cfg.llm.max_new_tokens,
        temperature=cfg.llm.temperature,
    )


def build_pipeline(cfg: Config, retriever=None, generator=None) -> RAGPipeline:
    g = cfg.guardrails
    return RAGPipeline(
        retriever or build_retriever(cfg),
        generator or build_generator(cfg),
        require_citations=g.require_citations,
        mask_pii=g.mask_pii,
        max_question_chars=g.max_question_chars,
        citation_repair=g.get("citation_repair", True),
    )
