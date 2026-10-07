"""PDF -> list[Page], one entry per physical page, numbered from 1."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

EMPTY_PAGE_CHARS = 20
PARAGRAPH_GAP_FACTOR = 1.4
_SPACES = re.compile(r"\s+")


class InvalidPDFError(ValueError):
    """The file is not a readable PDF (wrong format, corrupt, or password-protected)."""


@dataclass(frozen=True)
class Page:
    number: int
    text: str
    is_empty: bool


def load_pdf(path: str | Path) -> list[Page]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    with path.open("rb") as fh:
        if fh.read(5) != b"%PDF-":
            raise InvalidPDFError(f"{path.name} is not a PDF file")

    try:
        with pdfplumber.open(path) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages, start=1):
                text = _page_text(page)
                pages.append(Page(number=i, text=text, is_empty=len(text) < EMPTY_PAGE_CHARS))
    except Exception as exc:  # pdfminer raises many unrelated types for bad/encrypted input
        reason = (
            "it is password-protected"
            if _is_password_error(exc)
            else str(exc) or type(exc).__name__
        )
        raise InvalidPDFError(f"Cannot read {path.name}: {reason}") from exc
    return pages


def _is_password_error(exc: BaseException) -> bool:
    while exc is not None:
        if "Password" in type(exc).__name__:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _page_text(page) -> str:
    """Rebuild paragraphs from line geometry: a vertical gap wider than usual starts a new one."""
    lines = [(ln["top"], _SPACES.sub(" ", ln["text"]).strip()) for ln in page.extract_text_lines()]
    lines = [(top, text) for top, text in lines if text]
    if not lines:
        return ""

    steps = [b[0] - a[0] for a, b in zip(lines, lines[1:], strict=False) if b[0] > a[0]]
    usual = statistics.median(steps) if steps else 0.0

    paragraphs: list[str] = []
    current = lines[0][1]
    for (prev_top, _), (top, text) in zip(lines, lines[1:], strict=False):
        if usual and top - prev_top > usual * PARAGRAPH_GAP_FACTOR:
            paragraphs.append(current)
            current = text
        else:
            current = _join_line(current, text)
    paragraphs.append(current)
    return "\n\n".join(paragraphs)


def _join_line(left: str, right: str) -> str:
    """Join two wrapped lines, undoing a hyphen that only exists because of the line break."""
    if len(left) > 1 and left.endswith("-") and left[-2].isalpha() and right[:1].islower():
        return left[:-1] + right
    return f"{left} {right}"
