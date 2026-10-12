"""Compare pipeline variants on the evaluation set and log results to MLflow.

    python -m cia.eval.run_eval                          # all variants
    python -m cia.eval.run_eval --variants rag_dense rag_hybrid_rerank
    python -m cia.eval.run_eval --limit 50               # quick run

Variants (each is one ablation step, so you can see what each technique adds):
    prompt_only          no retrieval (baseline)
    rag_dense            dense vector search only
    rag_hybrid           dense + BM25 with reciprocal rank fusion
    rag_hybrid_rerank    hybrid + cross-encoder re-ranking
    rag_lora             hybrid + re-ranking + LoRA fine-tuned generator

Outputs results/eval_results.csv, results/eval_summary.md and per-question answers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

from cia.config import Config, load_config
from cia.eval.metrics import (
    groundedness,
    hallucinated,
    keyword_accuracy,
    recall_at_k,
    token_f1,
)

VARIANTS = {
    "prompt_only": {"mode": "prompt_only"},
    "rag_dense": {"mode": "rag", "hybrid": False, "rerank": False},
    "rag_hybrid": {"mode": "rag", "hybrid": True, "rerank": False},
    "rag_hybrid_rerank": {"mode": "rag", "hybrid": True, "rerank": True},
    "rag_lora": {"mode": "rag", "hybrid": True, "rerank": True, "adapter": True},
}


def load_questions(path: str | Path, limit: int | None = None) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return rows[:limit] if limit else rows


def evaluate(pipeline, questions: list[dict], variant: dict, threshold: float) -> dict:
    per_question = []
    latencies = []
    for q in questions:
        t0 = time.perf_counter()
        ans = pipeline.ask(
            q["question"],
            mode=variant["mode"],
            hybrid=variant.get("hybrid"),
            rerank=variant.get("rerank"),
        )
        latencies.append(time.perf_counter() - t0)
        contexts = [c.chunk.text for c in ans.contexts]
        sources = [c.chunk.source for c in ans.contexts]
        row = {
            "question": q["question"],
            "answer": ans.text,
            "accuracy": keyword_accuracy(ans.text, q.get("keywords", [])),
            "token_f1": token_f1(ans.text, q.get("reference", "")),
            "retrieval_recall": recall_at_k(sources, q.get("gold_sources", []))
            if variant["mode"] == "rag"
            else None,
        }
        if variant["mode"] == "rag":
            row["groundedness"] = groundedness(ans.text, contexts, threshold)
            row["hallucination"] = hallucinated(ans.text, contexts, threshold)
        per_question.append(row)

    def avg(key: str) -> float | None:
        vals = [r[key] for r in per_question if r.get(key) is not None]
        return round(mean(vals), 4) if vals else None

    summary = {
        "n": len(per_question),
        "accuracy": avg("accuracy"),
        "token_f1": avg("token_f1"),
        "groundedness": avg("groundedness"),
        "hallucination_rate": avg("hallucination"),
        "retrieval_recall@k": avg("retrieval_recall"),
        "p50_latency_s": round(sorted(latencies)[len(latencies) // 2], 3) if latencies else None,
    }
    return {"summary": summary, "rows": per_question}


def run(cfg: Config, variant_names: list[str], limit: int | None = None) -> dict:
    from cia.factory import build_generator, build_pipeline, build_retriever

    questions = load_questions(cfg.evaluation.questions_file, limit)
    threshold = cfg.evaluation.grounding_threshold
    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    retriever = build_retriever(cfg)
    base_generator = None
    tracker = _mlflow(cfg)
    summaries: dict[str, dict] = {}

    for name in variant_names:
        variant = VARIANTS[name]
        if variant.get("adapter"):
            generator = build_generator(cfg, use_adapter=True)
        else:
            base_generator = base_generator or build_generator(cfg, use_adapter=False)
            generator = base_generator
        pipeline = build_pipeline(cfg, retriever=retriever, generator=generator)

        result = evaluate(pipeline, questions, variant, threshold)
        summaries[name] = result["summary"]
        with open(results_dir / f"answers_{name}.jsonl", "w", encoding="utf-8") as fh:
            for row in result["rows"]:
                fh.write(json.dumps(row) + "\n")

        if tracker:
            with tracker.start_run(run_name=name):
                tracker.log_params({"variant": name, **{k: str(v) for k, v in variant.items()},
                                    "llm": cfg.llm.model, "embeddings": cfg.embeddings.model})
                tracker.log_metrics({k: v for k, v in result["summary"].items()
                                     if isinstance(v, int | float)})
        print(name, json.dumps(result["summary"]))

    _write_summary(summaries, results_dir)
    return summaries


def _mlflow(cfg: Config):
    try:
        import mlflow
    except ImportError:
        return None
    mlflow.set_experiment(cfg.evaluation.experiment_name)
    return mlflow


def _write_summary(summaries: dict[str, dict], results_dir: Path) -> None:
    cols = ["accuracy", "token_f1", "groundedness", "hallucination_rate",
            "retrieval_recall@k", "p50_latency_s", "n"]
    lines = [",".join(["variant", *cols])]
    md = ["| Variant | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, s in summaries.items():
        vals = ["" if s.get(c) is None else str(s[c]) for c in cols]
        lines.append(",".join([name, *vals]))
        md.append(f"| {name} | " + " | ".join(v or "–" for v in vals) + " |")
    (results_dir / "eval_results.csv").write_text("\n".join(lines) + "\n")
    (results_dir / "eval_summary.md").write_text("\n".join(md) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(load_config(args.config), args.variants, args.limit)


if __name__ == "__main__":
    main()
