"""pages -> overlapping chunks that remember which page(s) they came from.

The document is treated as one continuous text so a sentence that runs over a page break stays
in one chunk. Each chunk ends at the best boundary that fits in ``size`` characters, preferring
a paragraph break, then a sentence end, then a space, so words are never cut. The next chunk
starts at the first word inside the last ``overlap`` characters of the previous one.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from app.ingest.loader import Page

_SENTENCE_END = re.compile(r"[.!?][\"')\]]?$")
_PARAGRAPH, _SENTENCE, _WORD = 3, 2, 1


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    pages: list[int] = field(default_factory=list)
    start_char: int = 0  # offset of the chunk's first character within its first page's text

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(**data)


def chunk_pages(
    pages: list[Page], size: int = 800, overlap: int = 150, doc_id: str = ""
) -> list[Chunk]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")

    text, spans = _join_pages(pages)
    if not text:
        return []
    breaks = _break_points(text)

    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = len(text) if len(text) - start <= size else _choose_end(text, breaks, start, size)
        body = text[start:end].strip()
        touched = [n for n, s, e in spans if s < end and e > start]
        first_page_start = next(s for n, s, e in spans if n == touched[0])
        chunks.append(
            Chunk(
                id=f"{doc_id[:8]}-{len(chunks):04d}",
                text=body,
                pages=touched,
                start_char=max(start - first_page_start, 0),
            )
        )
        if end >= len(text):
            break
        start = _next_start(text, start, end, overlap)
    return chunks


def _join_pages(pages: list[Page]) -> tuple[str, list[tuple[int, int, int]]]:
    """Concatenate non-empty pages. A page that ends mid-sentence continues with a space."""
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    pos = 0
    for page in pages:
        if page.is_empty or not page.text.strip():
            continue
        if parts:
            sep = "\n\n" if _SENTENCE_END.search(parts[-1].rstrip()) else " "
            parts.append(sep)
            pos += len(sep)
        body = page.text.strip()
        spans.append((page.number, pos, pos + len(body)))
        parts.append(body)
        pos += len(body)
    return "".join(parts), spans


def _break_points(text: str) -> dict[int, int]:
    """Map each whitespace position where a chunk may end to the strength of that boundary."""
    points: dict[int, int] = {}
    for m in re.finditer(r"\s+", text):
        if "\n\n" in m.group():
            points[m.start()] = _PARAGRAPH
        elif _SENTENCE_END.search(text[max(m.start() - 3, 0) : m.start()]):
            points[m.start()] = _SENTENCE
        else:
            points[m.start()] = _WORD
    return points


def _choose_end(text: str, breaks: dict[int, int], start: int, size: int) -> int:
    limit = start + size
    floor = start + size // 2
    candidates = [p for p in breaks if floor <= p <= limit]
    if candidates:
        # strongest boundary wins; among equals, the latest keeps chunks full
        return max(candidates, key=lambda p: (breaks[p], p))
    later = [p for p in breaks if start < p <= limit]
    if later:
        return max(later)
    # a single "word" longer than the chunk size: end at the next whitespace instead of cutting it
    after = [p for p in breaks if p > start]
    return min(after) if after else len(text)


def _next_start(text: str, start: int, end: int, overlap: int) -> int:
    target = max(end - overlap, start + 1)
    while target < end and not (text[target - 1].isspace() and not text[target].isspace()):
        target += 1
    if target >= end:  # no word start inside the overlap window: begin after the break
        target = end
        while target < len(text) and text[target].isspace():
            target += 1
    return target
