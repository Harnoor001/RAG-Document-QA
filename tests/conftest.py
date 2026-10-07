import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
HANDBOOK = FIXTURES / "handbook.pdf"


@pytest.fixture(scope="session")
def handbook() -> Path:
    if not HANDBOOK.exists():  # the PDF is committed, but regenerate it if someone deleted it
        sys.path.insert(0, str(ROOT / "scripts"))
        import make_fixture_pdf

        make_fixture_pdf.build(FIXTURES)
    return HANDBOOK


@pytest.fixture(scope="session")
def pages(handbook):
    from app.ingest.loader import load_pdf

    return load_pdf(handbook)


@pytest.fixture
def make_pdf(tmp_path):
    """Write a small PDF whose pages hold the given strings."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    def _make(name: str, page_texts: list[str], **kwargs) -> Path:
        path = tmp_path / name
        c = canvas.Canvas(str(path), pagesize=LETTER, **kwargs)
        for text in page_texts:
            c.setFont("Helvetica", 11)
            c.drawString(72, 700, text)
            c.showPage()
        c.save()
        return path

    return _make
