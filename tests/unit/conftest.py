import hashlib
from pathlib import Path

import pytest

from app.index import store as store_module
from app.ingest.loader import load_pdf

_PARSED: dict[str, list] = {}


def cached_load_pdf(path):
    """pdfplumber needs ~1.3 s for the 42-page fixture. The loader has its own tests, so other
    tests reuse its parsed pages. Keyed by content: API uploads land in a new temp file."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest not in _PARSED:
        _PARSED[digest] = load_pdf(path)
    return _PARSED[digest]


@pytest.fixture(scope="session", autouse=True)
def fast_pdf_loading():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(store_module, "load_pdf", cached_load_pdf)
        yield


@pytest.fixture(scope="session")
def pages(handbook):
    return cached_load_pdf(handbook)
