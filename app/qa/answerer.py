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
_ONE_TAG = r"\[\s*C\d+(?:\s*[,;]\s*C\d+)*\s*\]"
_TAG_RUN = re.compile(rf"{_ONE_TAG}(?:[ \t]*{_ONE_TAG})*", re.IGNORECASE)  # [C1][C2], [C1, C2]
_TAG_NUMBER = re.compile(r"C(\d+)", re.IGNORECASE)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    pages: list[int]
    score: float
    snippet: str
    chunk_pages: list[int] = field(default_factory=list)  # every page the cited chunk spans


@dataclass
class Answer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    sources: int = 0
    refused: bool = False
    grounded: bool = False
    latency_ms: int = 0  # load_ms + answer_ms
    load_ms: int = 0  # one-off model/index loading paid by this call (embedder, FAISS, Ollama)
    answer_ms: int = 0  # retrieval + generation once everything is loaded
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
        self.retriever.warm_up(doc_id)
        timer = _Timer(started, load_s=time.perf_counter() - started)
        retrieval = self.retriever.retrieve(doc_id, question, k)

        if not retrieval.chunks or retrieval.top_score < self.threshold:
            return self._refusal(retrieval.top_score, timer)

        system, user = build_prompt(question, retrieval.chunks)
        raw = self.llm.generate(system, user)
        timer.load_s += getattr(self.llm, "last_load_s", 0.0)  # Ollama loading the model
        text, cited = _resolve_citations(raw, retrieval.chunks)

        if _is_refusal(raw) or not cited:
            return self._refusal(retrieval.top_score, timer)

        citations = [
            Citation(
                chunk_id=c.item.chunk.id,
                pages=sorted(c.pages),
                score=round(c.item.score, 4),
                snippet=_snippet(c.item.chunk.text),
                chunk_pages=list(c.item.chunk.pages),
            )
            for c in cited
        ]
        return Answer(
            answer=text,
            citations=citations,
            confidence=round(max(c.item.score for c in cited), 2),
            sources=len(citations),
            refused=False,
            grounded=True,
            model=self.llm.name,
            **timer.fields(),
        )

    def _refusal(self, top_score: float, timer: _Timer) -> Answer:
        return Answer(
            answer=REFUSAL,
            confidence=round(max(top_score, 0.0), 2),
            refused=True,
            grounded=False,
            model=self.llm.name,
            **timer.fields(),
        )


class _Timer:
    def __init__(self, started: float, load_s: float):
        self.started = started
        self.load_s = load_s

    def fields(self) -> dict[str, int]:
        total = time.perf_counter() - self.started
        load = min(self.load_s, total)
        return {
            "latency_ms": round(total * 1000),
            "load_ms": round(load * 1000),
            "answer_ms": round(total * 1000) - round(load * 1000),
        }


def _resolve_citations(raw: str, chunks: list[ScoredChunk]) -> tuple[str, list[_Cited]]:
    """Rewrite [C#] tags as [p.N]; drop tags that point outside the retrieved set.

    Each tag run is checked against the claim it follows (the text since the previous run):

    - A cited chunk the claim has no real overlap with is dropped, when another chunk in the
      same run clearly supports the claim (see ``_supporting``).
    - A cited chunk that spans pages is cited only for the page(s) holding the sentences that
      match the claim (see ``_supporting_pages``).

    Adjacent tags are rendered together, and their pages are merged when they overlap, so
    chunks on pages [1, 2] and [1] cited side by side read "[p.1-2]", not "[p.1-2][p.1]".
    """
    whole = _TAG_RUN.sub(" ", raw)
    cited: dict[str, _Cited] = {}
    parts: list[str] = []
    prev = 0
    for match in _TAG_RUN.finditer(raw):
        claim = raw[prev : match.start()]
        parts.append(claim)
        prev = match.end()
        run: list[ScoredChunk] = []
        for number in _TAG_NUMBER.findall(match.group(0)):
            index = int(number) - 1
            if 0 <= index < len(chunks) and chunks[index] not in run:
                run.append(chunks[index])  # out-of-range numbers are hallucinated blocks
        page_sets: list[set[int]] = []
        for item in _supporting(claim, run, whole):
            pages = _supporting_pages(claim, item.chunk)
            entry = cited.setdefault(item.chunk.id, _Cited(item, set()))
            entry.pages |= pages
            page_sets.append(pages)
        parts.append(
            "".join(f"[p.{format_pages(sorted(p))}]" for p in _merge_overlapping(page_sets))
        )
    parts.append(raw[prev:])
    text = "".join(parts)
    text = re.sub(r"(\[p\.[^\]]+\])(?:\1)+", r"\1", text)  # [p.12][p.12] -> [p.12]
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()
    return text, list(cited.values())


@dataclass
class _Cited:
    item: ScoredChunk
    pages: set[int]


_STOPWORDS = frozenset(
    [
        "the",
        "and",
        "for",
        "are",
        "was",
        "were",
        "with",
        "that",
        "this",
        "from",
        "have",
        "has",
        "had",
        "not",
        "but",
        "any",
        "all",
        "can",
        "may",
        "shall",
        "will",
        "would",
        "should",
        "could",
        "been",
        "being",
        "its",
        "their",
        "them",
        "they",
        "there",
        "then",
        "than",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "when",
        "where",
        "how",
        "why",
        "into",
        "onto",
        "upon",
        "such",
        "each",
        "other",
        "these",
        "those",
        "also",
        "only",
        "more",
        "most",
        "some",
        "per",
        "our",
        "your",
        "his",
        "her",
        "its",
        "out",
        "under",
        "over",
        "about",
        "after",
        "before",
        "between",
        "does",
        "did",
        "done",
        "must",
        "very",
        "just",
    ]
)
_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")


def _terms(text: str) -> set[str]:
    """Content words, crudely stemmed by truncation so "ratified" matches "ratification"."""
    words = _WORD.findall(text.lower())
    return {
        w if w.isdigit() else w[:6]
        for w in words
        if w not in _STOPWORDS and (len(w) >= 3 or w.isdigit())
    }


def _supporting(claim: str, run: list[ScoredChunk], whole: str) -> list[ScoredChunk]:
    """Chunks in one tag run that actually support the claim before it.

    A chunk is dropped when it shares no content word with the claim, or fewer than half as
    many as the best-matching chunk in the run. When no chunk in the run overlaps the claim
    (say the tag sits after a one-word sentence), the whole answer is used as the claim. If
    nothing overlaps even then there is no evidence to choose by, so every cited chunk is kept.
    """
    for text in (claim, whole):
        claim_terms = _terms(text)
        shared = [len(claim_terms & _terms(item.chunk.text)) for item in run]
        best = max(shared, default=0)
        if best > 0:
            break
    else:
        return run
    return [item for item, n in zip(run, shared, strict=True) if n > 0 and n >= best / 2]


def _supporting_pages(claim: str, chunk) -> set[int]:
    """The page(s) of ``chunk`` holding the sentences that support the claim.

    The chunk sentence sharing the most content words with the claim is taken first; further
    sentences are added only while each brings at least two claim words not yet covered, so a
    claim that combines facts from both pages cites both, and a passing shared word does not.
    A sentence that runs over a page break counts for both pages. Chunks without page offsets
    (indexes built before they existed) keep all their pages.
    """
    pages = list(chunk.pages)
    starts = list(getattr(chunk, "page_starts", []) or [])
    if len(pages) < 2 or len(starts) != len(pages):
        return set(pages)
    claim_terms = _terms(claim)
    sentences = []  # (start, end, claim terms in the sentence)
    pos = 0
    for piece in _SENTENCE_SPLIT.split(chunk.text):
        begin = chunk.text.find(piece, pos)
        pos = begin + len(piece)
        if piece.strip():
            sentences.append((begin, pos, claim_terms & _terms(piece)))
    covered: set[str] = set()
    picked = []
    while True:
        best = max(sentences, key=lambda x: len(x[2] - covered), default=None)
        gain = len(best[2] - covered) if best else 0
        if gain == 0 or (picked and gain < 2):
            break
        picked.append(best)
        covered |= best[2]
    if not picked:
        return set(pages)
    bounds = [*starts[1:], len(chunk.text) + 1]
    return {
        page
        for begin, end, _ in picked
        for page, lo, hi in zip(pages, starts, bounds, strict=True)
        if begin < hi and end > lo
    }


def _merge_overlapping(page_sets: list[set[int]]) -> list[set[int]]:
    """Union page sets that share a page, keeping first-appearance order."""
    merged: list[set[int]] = []
    for pages in page_sets:
        pages = set(pages)
        overlapping = [m for m in merged if m & pages]
        for m in overlapping:
            pages |= m
        position = merged.index(overlapping[0]) if overlapping else len(merged)
        merged = [m for m in merged if m not in overlapping]
        merged.insert(min(position, len(merged)), pages)
    return merged


def _is_refusal(raw: str) -> bool:
    norm = raw.replace("’", "'").lower()
    return REFUSAL.lower().rstrip(".") in norm


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= SNIPPET_CHARS:
        return flat
    return flat[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
