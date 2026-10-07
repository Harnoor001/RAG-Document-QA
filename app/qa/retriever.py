"""Top-k search over one document, with near-duplicate removal and a context budget."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from app.config import get_settings
from app.ingest.chunker import Chunk

DUPLICATE_JACCARD = 0.8
_WORD = re.compile(r"\w+")


class Searchable(Protocol):
    def search(self, doc_id: str, query: str, k: int = 5) -> list[tuple[Chunk, float]]: ...


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass
class RetrievalResult:
    chunks: list[ScoredChunk] = field(default_factory=list)  # best first
    top_score: float = 0.0  # best score before any filtering, used by the confidence gate


class Retriever:
    def __init__(
        self, store: Searchable, top_k: int | None = None, max_context_chars: int | None = None
    ):
        settings = get_settings()
        self.store = store
        self.top_k = top_k if top_k is not None else settings.top_k
        self.max_context_chars = (
            max_context_chars if max_context_chars is not None else settings.max_context_chars
        )

    def warm_up(self, doc_id: str) -> None:
        """Load the document index and the embedding model, so their cost is timed apart."""
        warm = getattr(self.store, "warm_up", None)
        if warm is not None:
            warm(doc_id)

    def retrieve(self, doc_id: str, question: str, k: int | None = None) -> RetrievalResult:
        k = k if k is not None else self.top_k
        # over-fetch so that dropping duplicates still leaves k distinct chunks
        hits = self.store.search(doc_id, question, k * 2)
        if not hits:
            return RetrievalResult()
        top_score = hits[0][1]
        kept = _dedupe([ScoredChunk(c, s) for c, s in hits])[:k]
        return RetrievalResult(_fit_budget(kept, self.max_context_chars), top_score)


def _dedupe(ranked: list[ScoredChunk]) -> list[ScoredChunk]:
    """Drop a chunk that is (nearly) the same text as a higher-scoring one."""
    kept: list[tuple[ScoredChunk, set[str]]] = []
    for item in ranked:
        words = set(_WORD.findall(item.chunk.text.lower()))
        if not any(_is_duplicate(item.chunk.text, words, k.chunk.text, w) for k, w in kept):
            kept.append((item, words))
    return [item for item, _ in kept]


def _is_duplicate(text_a: str, words_a: set[str], text_b: str, words_b: set[str]) -> bool:
    if text_a in text_b or text_b in text_a:
        return True
    union = words_a | words_b
    return bool(union) and len(words_a & words_b) / len(union) >= DUPLICATE_JACCARD


def _fit_budget(ranked: list[ScoredChunk], budget: int) -> list[ScoredChunk]:
    """Keep the best chunks whose text fits in ``budget`` chars; the best one always stays."""
    kept: list[ScoredChunk] = []
    used = 0
    for item in ranked:
        size = len(item.chunk.text)
        if kept and used + size > budget:
            break
        kept.append(item)
        used += size
    return kept
