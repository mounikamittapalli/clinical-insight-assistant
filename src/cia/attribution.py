"""Citation repair (post-hoc attribution).

Small models often answer correctly from the context but forget the [n] markers.
Instead of throwing those answers away, match each answer sentence to the retrieved
passage that best supports it and add the citation. Sentences no passage supports are
dropped; if nothing is supported, the caller treats the answer as ungrounded.
"""

from __future__ import annotations

import re

from cia.text import split_sentences, tokenize

_CITE = re.compile(r"\s*\[\d+\]")


def support(sentence: str, passage_tokens: set[str]) -> float:
    """Share of the sentence's content words that appear in the passage."""
    tokens = tokenize(_CITE.sub("", sentence))
    if not tokens:
        return 0.0
    return sum(t in passage_tokens for t in tokens) / len(tokens)


def verify_citations(answer: str, passages: list[str], threshold: float = 0.6) -> str | None:
    """Check every sentence against the passage(s) it cites.

    - Cited and supported by its cited passage(s): kept as is.
    - Uncited, or cited to a passage that doesn't support it: re-cited to the best
      supporting passage if one clears the threshold.
    - Supported by no passage: dropped (it came from the model's own knowledge or was
      made up, so it must not be presented as sourced).

    Returns the verified answer, or None if no sentence is supported.
    """
    if not passages:
        return None
    passage_tokens = [set(tokenize(p)) for p in passages]
    kept: list[str] = []
    for sentence in split_sentences(answer):
        cited = [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]
        valid = [n for n in cited if 1 <= n <= len(passages)]
        if valid and all(support(sentence, passage_tokens[n - 1]) >= threshold for n in valid):
            kept.append(sentence)
            continue
        scores = [support(sentence, toks) for toks in passage_tokens]
        best = max(range(len(scores)), key=scores.__getitem__)
        if scores[best] < threshold:
            continue
        body = _CITE.sub("", sentence).rstrip()
        end = body[-1] if body and body[-1] in ".!?" else "."
        kept.append(f"{body.rstrip('.!?')} [{best + 1}]{end}")
    return " ".join(kept) if kept else None


def add_citations(answer: str, passages: list[str], threshold: float = 0.6) -> str | None:
    """Return the answer with [n] added to each supported sentence, or None if no
    sentence is supported by any passage."""
    if not passages:
        return None
    passage_tokens = [set(tokenize(p)) for p in passages]
    cited: list[str] = []
    for sentence in split_sentences(answer):
        scores = [support(sentence, toks) for toks in passage_tokens]
        best = max(range(len(scores)), key=scores.__getitem__)
        if scores[best] < threshold:
            continue  # unsupported sentence: leave it out rather than cite it falsely
        body = _CITE.sub("", sentence).rstrip()
        end = body[-1] if body and body[-1] in ".!?" else "."
        body = body.rstrip(".!?")
        cited.append(f"{body} [{best + 1}]{end}")
    return " ".join(cited) if cited else None
