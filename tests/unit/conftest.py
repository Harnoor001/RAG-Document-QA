from functools import lru_cache
from pathlib import Path

import pytest

from app.index import store as store_module
from app.ingest.loader import load_pdf


@lru_cache(maxsize=8)
def _cached_load(path: str, size: int, mtime: float):
    return load_pdf(path)


@pytest.fixture(scope="session", autouse=True)
def fast_pdf_loading():
    """pdfplumber needs ~1.3 s for the 42-page fixture. The loader has its own tests, so the
    store tests reuse its parsed pages instead of re-parsing the same file in every test."""

    def load(path):
        stat = Path(path).stat()
        return _cached_load(str(Path(path).resolve()), stat.st_size, stat.st_mtime)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(store_module, "load_pdf", load)
        yield


@pytest.fixture(scope="session")
def pages(handbook):
    stat = handbook.stat()
    return _cached_load(str(handbook.resolve()), stat.st_size, stat.st_mtime)
