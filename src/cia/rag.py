"""The RAG pipeline: guardrails -> retrieve -> prompt -> generate -> guardrails."""

from __future__ import annotations

from cia.attribution import verify_citations
from cia.guardrails import NO_ANSWER, check_input, check_output, cited_indices
from cia.llm import Generator
from cia.prompts import plain_messages, rag_messages
from cia.retrieval import HybridRetriever
from cia.schema import Answer


class RAGPipeline:
    MODES = ("rag", "prompt_only")

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: Generator,
        require_citations: bool = True,
        mask_pii: bool = True,
        max_question_chars: int = 1000,
        citation_repair: bool = True,
        repair_threshold: float = 0.6,
    ):
        self.citation_repair = citation_repair
        self.repair_threshold = repair_threshold
        self.retriever = retriever
        self.generator = generator
        self.require_citations = require_citations
        self.mask_pii = mask_pii
        self.max_question_chars = max_question_chars

    def ask(
        self,
        question: str,
        mode: str = "rag",
        hybrid: bool | None = None,
        rerank: bool | None = None,
        top_k: int | None = None,
    ) -> Answer:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")

        checked = check_input(question, self.max_question_chars)
        if not checked.allowed:
            return Answer(question, NO_ANSWER, [], [], mode, blocked=True, reason=checked.reason)
        q = checked.text

        if mode == "prompt_only":
            raw = self.generator.generate(plain_messages(q))
            out = check_output(raw, 0, require_citations=False, mask=self.mask_pii)
            return Answer(question, out.text, [], [], mode)

        contexts = self.retriever.retrieve(q, top_k=top_k, hybrid=hybrid, rerank=rerank)
        raw = self.generator.generate(rag_messages(q, contexts))
        if (
            self.citation_repair
            and self.require_citations
            and contexts
            and NO_ANSWER.lower()[:20] not in raw.lower()
        ):
            # Verify every sentence against the passage it cites; re-cite or drop
            # sentences the sources don't support.
            verified = verify_citations(
                raw, [c.chunk.text for c in contexts], self.repair_threshold
            )
            if verified is None:
                return Answer(
                    question, NO_ANSWER, [], contexts, mode,
                    blocked=True, reason="answer not supported by the retrieved sources",
                )
            raw = verified
        out = check_output(raw, len(contexts), self.require_citations, mask=self.mask_pii)
        citations = [contexts[i - 1] for i in cited_indices(out.text) if 1 <= i <= len(contexts)]
        return Answer(
            question,
            out.text,
            citations,
            contexts,
            mode,
            blocked=not out.allowed,
            reason=out.reason,
        )
