import json
from pathlib import Path

from cia.build_index import build_index
from cia.eval.run_eval import VARIANTS, evaluate, load_questions
from cia.factory import build_pipeline, build_retriever
from cia.finetune.prepare_data import prepare


def test_prepare_data_splits_patients(cfg, embedder):
    build_index(cfg, embedder=embedder)
    stats = prepare(cfg, eval_fraction=0.34)
    assert stats["eval_patients"] == 1 and stats["train_patients"] == 2

    train = [json.loads(line) for line in open(Path(cfg.paths.finetune_dir) / "train.jsonl")]
    example = train[0]["messages"]
    assert [m["role"] for m in example] == ["system", "user", "assistant"]
    assert "Context passages" in example[1]["content"]

    questions = load_questions(cfg.evaluation.questions_file)
    assert questions and all("gold_sources" in q for q in questions)
    eval_patients = {q["gold_sources"][0] for q in questions}
    train_text = " ".join(m["content"] for ex in train for m in ex["messages"])
    for source in eval_patients:  # no held-out patient appears in training data
        assert source.split(":")[1] not in train_text


def test_evaluate_reports_metrics(cfg, embedder, fake_llm):
    build_index(cfg, embedder=embedder)
    pipeline = build_pipeline(
        cfg, retriever=build_retriever(cfg, embedder=embedder), generator=fake_llm
    )
    questions = [{
        "question": "What active conditions does patient p-001 have?",
        "reference": "Diabetes mellitus type 2; Essential hypertension [1].",
        "keywords": ["p-001"],
        "gold_sources": ["patient:p-001"],
    }]
    result = evaluate(pipeline, questions, VARIANTS["rag_hybrid"], threshold=0.55)
    summary = result["summary"]
    assert summary["retrieval_recall@k"] == 1.0
    assert summary["accuracy"] == 1.0
    assert summary["hallucination_rate"] == 0.0
    baseline = evaluate(pipeline, questions, VARIANTS["prompt_only"], threshold=0.55)
    assert baseline["summary"]["groundedness"] is None
