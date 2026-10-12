"""Input and output guardrails.

- PII masking: redacts identifiers (SSN, phone, email, MRN-style IDs, street addresses)
  before text reaches the LLM and before answers leave the API.
- Input checks: length limits and common prompt-injection phrasings.
- Output checks: answers must cite retrieved sources, or say the context doesn't cover it.

Synthea data is synthetic, but the pipeline is written as if it weren't.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PII_PATTERNS: dict[str, re.Pattern] = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"),
    "MRN": re.compile(r"\bMRN[:#\s]*\d{5,}\b", re.IGNORECASE),
    "ADDRESS": re.compile(
        r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|"
        r"Drive|Dr|Boulevard|Blvd|Court|Ct)\b\.?"
    ),
}

INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore (all |any )?(previous|prior|above) instructions",
        r"disregard (the )?(system|previous) prompt",
        r"reveal (your|the) (system )?prompt",
        r"you are now (in )?developer mode",
    ]
]

NO_ANSWER = "I don't know based on the available records and policy documents."
_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class GuardrailResult:
    allowed: bool
    text: str
    reason: str | None = None


def mask_pii(text: str) -> str:
    for label, pattern in PII_PATTERNS.items():
        text = pattern.sub(f"[{label}]", text)
    return text


def check_input(question: str, max_chars: int = 1000) -> GuardrailResult:
    q = question.strip()
    if not q:
        return GuardrailResult(False, q, "empty question")
    if len(q) > max_chars:
        return GuardrailResult(False, q, f"question longer than {max_chars} characters")
    if any(p.search(q) for p in INJECTION_PATTERNS):
        return GuardrailResult(False, q, "possible prompt injection")
    return GuardrailResult(True, mask_pii(q))


def cited_indices(answer: str) -> list[int]:
    return sorted({int(m) for m in _CITATION.findall(answer)})


def check_output(
    answer: str, n_contexts: int, require_citations: bool = True, mask: bool = True
) -> GuardrailResult:
    text = mask_pii(answer) if mask else answer
    if not require_citations or n_contexts == 0:
        return GuardrailResult(True, text)
    if NO_ANSWER.lower()[:20] in text.lower():
        return GuardrailResult(True, text)
    cites = cited_indices(text)
    if not cites:
        return GuardrailResult(False, NO_ANSWER, "answer had no citations")
    if any(c < 1 or c > n_contexts for c in cites):
        return GuardrailResult(False, NO_ANSWER, "answer cited a source that was not retrieved")
    return GuardrailResult(True, text)
