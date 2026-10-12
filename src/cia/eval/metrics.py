"""Evaluation metrics for RAG answers.

- accuracy: the answer contains all expected key facts (case-insensitive), or correctly
  refuses when no facts are expected.
- token_f1: token-overlap F1 against a reference answer.
- groundedness: share of answer sentences supported by the retrieved context.
- hallucination: 1 if any answer sentence is unsupported by the context.
- recall@k: whether a gold source was among the retrieved chunks.

Groundedness here is lexical (token coverage), so it needs no extra model and is
deterministic. Swap in an NLI or LLM-judge scorer for stricter checks.
"""

from __future__ import annotations

import re
from collections import Counter

from cia.guardrails import NO_ANSWER
from cia.text import split_sentences, tokenize

_CITE = re.compile(r"\[\d+\]")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_refusal(answer: str) -> bool:
    return _norm(NO_ANSWER)[:25] in _norm(answer) or "i don't know" in _norm(answer)


def keyword_accuracy(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0 if is_refusal(answer) else 0.0
    text = _norm(answer)
    return 1.0 if all(_norm(k) in text for k in keywords) else 0.0


def token_f1(answer: str, reference: str) -> float:
    pred, gold = tokenize(_CITE.sub("", answer)), tokenize(_CITE.sub("", reference))
    if not pred or not gold:
        return float(pred == gold)
    common = sum((Counter(pred) & Counter(gold)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def sentence_support(sentence: str, context_tokens: set[str]) -> float:
    tokens = tokenize(_CITE.sub("", sentence))
    if not tokens:
        return 1.0
    return sum(t in context_tokens for t in tokens) / len(tokens)


def groundedness(answer: str, contexts: list[str], threshold: float = 0.55) -> float:
    """Fraction of answer sentences whose tokens are mostly found in the context."""
    if is_refusal(answer):
        return 1.0
    sentences = split_sentences(answer)
    if not sentences:
        return 1.0
    ctx = set(tokenize(" ".join(contexts)))
    supported = [sentence_support(s, ctx) >= threshold for s in sentences]
    return sum(supported) / len(supported)


def hallucinated(answer: str, contexts: list[str], threshold: float = 0.55) -> float:
    return 0.0 if groundedness(answer, contexts, threshold) == 1.0 else 1.0


def recall_at_k(retrieved_sources: list[str], gold_sources: list[str]) -> float:
    if not gold_sources:
        return 1.0
    return 1.0 if any(g in retrieved_sources for g in gold_sources) else 0.0
