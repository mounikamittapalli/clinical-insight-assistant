"""Prompt templates. Kept in one place so prompt changes are easy to version and evaluate."""

from __future__ import annotations

from cia.guardrails import NO_ANSWER
from cia.schema import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a clinical and coverage-policy assistant for healthcare analysts. "
    "Answer ONLY from the numbered context passages. "
    "Cite every factual sentence with the passage number in square brackets, like [1] or [2][3]. "
    f'If the passages do not contain the answer, reply exactly: "{NO_ANSWER}" '
    "Be concise. Never invent patient details, codes or policy rules. "
    "Do not give medical advice."
)

PLAIN_SYSTEM_PROMPT = (
    "You are a clinical and coverage-policy assistant for healthcare analysts. "
    "Answer concisely. If you are not sure, say you don't know."
)


def format_context(contexts: list[RetrievedChunk], max_chars: int = 1200) -> str:
    blocks = []
    for i, item in enumerate(contexts, start=1):
        text = item.chunk.text[:max_chars]
        blocks.append(f"[{i}] (source: {item.chunk.source})\n{text}")
    return "\n\n".join(blocks)


CITATION_REMINDER = (
    "Answer in one to three short sentences using only the passages above. "
    "End every sentence with the number of the passage it came from, for example: "
    '"Patient 12ab has essential hypertension [1]."'
)


def rag_messages(question: str, contexts: list[RetrievedChunk]) -> list[dict]:
    # The citation rule is repeated right before the question: small models follow
    # instructions near the end of the prompt much more reliably.
    user = (
        f"Context passages:\n\n{format_context(contexts)}\n\n"
        f"{CITATION_REMINDER}\n\nQuestion: {question}\nAnswer:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def plain_messages(question: str) -> list[dict]:
    """Prompt-only baseline: no retrieval."""
    return [
        {"role": "system", "content": PLAIN_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\nAnswer:"},
    ]
