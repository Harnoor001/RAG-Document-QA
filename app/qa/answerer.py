"""retrieve -> confidence gate -> prompt -> generate -> validate citations -> Answer."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from app.config import get_settings
from app.qa.llm import LLM
from app.qa.prompt import REFUSAL, build_prompt, format_pages
from app.qa.retriever import Retriever, ScoredChunk

SNIPPET_CHARS = 200
_TAG = re.compile(r"\[\s*(C\d+(?:\s*[,;]\s*C\d+)*)\s*\]", re.IGNORECASE)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    pages: list[int]
    score: float
    snippet: str


@dataclass
class Answer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    sources: int = 0
    refused: bool = False
    grounded: bool = False
    latency_ms: int = 0
    model: str = ""


class Answerer:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        threshold: float | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.threshold = threshold if threshold is not None else get_settings().confidence_threshold

    def answer(self, doc_id: str, question: str, k: int | None = None) -> Answer:
        started = time.perf_counter()
        retrieval = self.retriever.retrieve(doc_id, question, k)

        if not retrieval.chunks or retrieval.top_score < self.threshold:
            return self._refusal(retrieval.top_score, started)

        system, user = build_prompt(question, retrieval.chunks)
        raw = self.llm.generate(system, user)
        text, cited = _resolve_citations(raw, retrieval.chunks)

        if _is_refusal(raw) or not cited:
            return self._refusal(retrieval.top_score, started)

        citations = [
            Citation(
                chunk_id=item.chunk.id,
                pages=list(item.chunk.pages),
                score=round(item.score, 4),
                snippet=_snippet(item.chunk.text),
            )
            for item in cited
        ]
        return Answer(
            answer=text,
            citations=citations,
            confidence=round(max(item.score for item in cited), 2),
            sources=len(citations),
            refused=False,
            grounded=True,
            latency_ms=_elapsed_ms(started),
            model=self.llm.name,
        )

    def _refusal(self, top_score: float, started: float) -> Answer:
        return Answer(
            answer=REFUSAL,
            confidence=round(max(top_score, 0.0), 2),
            refused=True,
            grounded=False,
            latency_ms=_elapsed_ms(started),
            model=self.llm.name,
        )


def _resolve_citations(raw: str, chunks: list[ScoredChunk]) -> tuple[str, list[ScoredChunk]]:
    """Rewrite [C#] tags as [p.N]; drop tags that point outside the retrieved set."""
    cited: list[ScoredChunk] = []

    def replace(match: re.Match) -> str:
        labels = []
        for tag in re.split(r"\s*[,;]\s*", match.group(1)):
            index = int(tag[1:]) - 1
            if not 0 <= index < len(chunks):
                continue  # hallucinated block number
            item = chunks[index]
            if item not in cited:
                cited.append(item)
            label = f"[p.{format_pages(item.chunk.pages)}]"
            if label not in labels:
                labels.append(label)
        return "".join(labels)

    text = _TAG.sub(replace, raw)
    text = re.sub(r"(\[p\.[^\]]+\])(?:\1)+", r"\1", text)  # [p.12][p.12] -> [p.12]
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()
    return text, cited


def _is_refusal(raw: str) -> bool:
    norm = raw.replace("’", "'").lower()
    return REFUSAL.lower().rstrip(".") in norm


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= SNIPPET_CHARS:
        return flat
    return flat[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
