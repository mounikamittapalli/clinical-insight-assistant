"""Build the retrieval index.

    python -m cia.build_index                 # uses configs/config.yaml
    python -m cia.build_index --config my.yaml

Steps: load Synthea CSVs -> write SQLite -> build patient summaries -> load CMS policy
documents -> chunk -> embed -> save the vector store, plus chunks for BM25.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cia.chunking import chunk_documents
from cia.config import Config, load_config
from cia.ingest.documents import load_policy_documents
from cia.ingest.synthea import load_tables, patient_documents, write_sqlite
from cia.vectorstore import build_store


def build_index(cfg: Config, embedder=None) -> dict:
    start = time.time()
    paths = cfg.paths

    frames = load_tables(paths.synthea_dir, cfg.ingest.get("max_patients"))
    write_sqlite(frames, paths.sqlite_db)
    patient_docs = patient_documents(frames)
    policy_docs = load_policy_documents(paths.policy_dir)
    docs = patient_docs + policy_docs
    if not docs:
        raise SystemExit(
            "No documents found. Run scripts/download_data.sh and add CMS policy files first."
        )

    chunks = chunk_documents(docs, cfg.chunking.chunk_size, cfg.chunking.chunk_overlap)

    if embedder is None:
        from cia.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(cfg.embeddings.model, cfg.embeddings.batch_size)
    vectors = embedder.embed([c.text for c in chunks])

    index_dir = Path(paths.index_dir)
    backend = cfg.vector_store.backend
    store = build_store(backend, embedder.dim, index_dir)
    store.add(chunks, vectors)
    store.save(index_dir)

    # BM25 is rebuilt from chunks at load time, so FAISS and Chroma share one chunk file.
    from cia.vectorstore import _save_chunks

    index_dir.mkdir(parents=True, exist_ok=True)
    _save_chunks(chunks, index_dir / "chunks.jsonl")

    stats = {
        "patients": len(patient_docs),
        "policy_documents": len(policy_docs),
        "chunks": len(chunks),
        "embedding_dim": embedder.dim,
        "backend": backend,
        "seconds": round(time.time() - start, 1),
    }
    (index_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    stats = build_index(load_config(args.config))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
