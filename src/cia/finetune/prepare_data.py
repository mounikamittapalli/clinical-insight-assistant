"""Generate grounded question-answer pairs from Synthea for LoRA fine-tuning and evaluation.

Every example is in the same RAG format the pipeline uses at inference time:
system prompt + numbered context passage(s) + question -> cited answer. The model learns to
answer from context, cite [n], and refuse when the context doesn't contain the answer.

Patients are split so that fine-tuning examples and evaluation questions never share
a patient (no train/test leakage).

    python -m cia.finetune.prepare_data
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path

import pandas as pd

from cia.config import Config, load_config
from cia.guardrails import NO_ANSWER
from cia.ingest.synthea import patient_documents
from cia.prompts import rag_messages
from cia.schema import Chunk, RetrievedChunk


def _frames_from_sqlite(db_path: str | Path) -> dict[str, pd.DataFrame]:
    with sqlite3.connect(db_path) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {t: pd.read_sql(f"SELECT * FROM {t}", conn) for t in tables}


def _active(df: pd.DataFrame) -> pd.DataFrame:
    if "STOP" not in df.columns:
        return df
    return df[df["STOP"].isna() | (df["STOP"].astype(str).str.strip() == "")]


def _join(items: list[str], limit: int = 6) -> str:
    items = list(dict.fromkeys(items))[:limit]
    return "; ".join(items)


def qa_for_patient(pid: str, frames: dict[str, pd.DataFrame]) -> list[dict]:
    """Template questions whose answers are known exactly from the structured data."""
    def rows(table: str) -> pd.DataFrame:
        df = frames.get(table, pd.DataFrame())
        return df[df["PATIENT"] == pid] if "PATIENT" in df.columns else df.iloc[0:0]

    qas: list[dict] = []
    conditions = list(_active(rows("conditions"))["DESCRIPTION"].dropna())
    if conditions:
        qas.append({
            "question": f"What active conditions does patient {pid} have?",
            "answer": f"Patient {pid} has these active conditions: {_join(conditions)} [1].",
            "keywords": list(dict.fromkeys(conditions))[:3],
        })
    meds = list(_active(rows("medications"))["DESCRIPTION"].dropna())
    if meds:
        qas.append({
            "question": f"Which medications is patient {pid} currently taking?",
            "answer": f"Patient {pid} is currently taking: {_join(meds)} [1].",
            "keywords": list(dict.fromkeys(meds))[:3],
        })
    allergies = list(rows("allergies")["DESCRIPTION"].dropna())
    if allergies:
        qas.append({
            "question": f"Does patient {pid} have any recorded allergies?",
            "answer": f"Yes. Patient {pid} has these recorded allergies: {_join(allergies)} [1].",
            "keywords": list(dict.fromkeys(allergies))[:3],
        })
    enc = rows("encounters")
    if len(enc):
        counts = enc["ENCOUNTERCLASS"].value_counts()
        cls, n = counts.index[0], int(counts.iloc[0])
        qas.append({
            "question": f"How many {cls} encounters has patient {pid} had?",
            "answer": f"Patient {pid} has had {n} {cls} encounters [1].",
            "keywords": [str(n)],
        })
    return qas


def _example(question: str, context: Chunk, answer: str) -> dict:
    messages = rag_messages(question, [RetrievedChunk(context, 1.0)])
    return {"messages": messages + [{"role": "assistant", "content": answer}]}


def prepare(cfg: Config, eval_fraction: float = 0.2, seed: int = 42) -> dict:
    rng = random.Random(seed)
    frames = _frames_from_sqlite(cfg.paths.sqlite_db)
    docs = {d.metadata["patient_id"]: d for d in patient_documents(frames)}
    pids = sorted(docs)
    rng.shuffle(pids)
    n_eval = max(1, int(len(pids) * eval_fraction))
    eval_pids, train_pids = pids[:n_eval], pids[n_eval:]

    def as_chunk(pid: str) -> Chunk:
        d = docs[pid]
        return Chunk(f"{d.doc_id}::0", d.doc_id, d.text, d.source, d.metadata)

    train: list[dict] = []
    for pid in train_pids:
        for qa in qa_for_patient(pid, frames):
            train.append(_example(qa["question"], as_chunk(pid), qa["answer"]))
            # Negative example: wrong patient's context -> the model must refuse.
            if rng.random() < 0.25 and len(train_pids) > 1:
                other = rng.choice([p for p in train_pids if p != pid])
                train.append(_example(qa["question"], as_chunk(other), NO_ANSWER))
    rng.shuffle(train)
    n_val = max(1, int(len(train) * 0.1))
    val, train = train[:n_val], train[n_val:]

    eval_rows = []
    for pid in eval_pids:
        for qa in qa_for_patient(pid, frames):
            eval_rows.append({
                "question": qa["question"],
                "reference": qa["answer"],
                "keywords": qa["keywords"],
                "gold_sources": [docs[pid].source],
            })

    out_dir = Path(cfg.paths.finetune_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "val.jsonl", val)
    eval_path = Path(cfg.evaluation.questions_file)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(eval_path, eval_rows)

    return {
        "train_examples": len(train),
        "val_examples": len(val),
        "eval_questions": len(eval_rows),
        "train_patients": len(train_pids),
        "eval_patients": len(eval_pids),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    print(json.dumps(prepare(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
