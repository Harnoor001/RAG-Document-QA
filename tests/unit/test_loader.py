import pytest

from app.ingest.loader import InvalidPDFError, _join_line, load_pdf


def test_t1_1_fixture_has_42_pages_numbered_from_1(pages):
    assert len(pages) == 42
    assert [p.number for p in pages] == list(range(1, 43))


def test_t1_2_page_12_contains_refund_window(pages):
    assert "30 days of purchase" in pages[11].text


def test_t1_3_page_33_is_flagged_empty(pages):
    assert pages[32].is_empty
    assert [p.number for p in pages if p.is_empty] == [33]


def test_t1_3_non_pdf_raises_invalid_pdf(tmp_path):
    fake = tmp_path / "notes.pdf"
    fake.write_text("just some text, not a PDF")
    with pytest.raises(InvalidPDFError, match="not a PDF"):
        load_pdf(fake)


def test_t1_3_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "nope.pdf")


def test_t1_3_corrupt_pdf_raises_invalid_pdf(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4\n garbage without any objects")
    with pytest.raises(InvalidPDFError):
        load_pdf(broken)


def test_t1_3_encrypted_pdf_raises_invalid_pdf(make_pdf):
    from reportlab.lib.pdfencrypt import StandardEncryption

    locked = make_pdf("locked.pdf", ["secret page text here"], encrypt=StandardEncryption("pw"))
    with pytest.raises(InvalidPDFError, match="password"):
        load_pdf(locked)


def test_whitespace_is_normalised(pages):
    for page in pages:
        assert "  " not in page.text
        assert "\n\n\n" not in page.text
        assert page.text == page.text.strip()


def test_paragraph_breaks_are_kept(pages):
    assert pages[11].text.count("\n\n") >= 3


def test_line_break_hyphens_are_removed():
    assert _join_line("an employ-", "ee handbook") == "an employee handbook"
    assert _join_line("form WC-", "2 is") == "form WC- 2 is"  # not a word split: keep it
    assert _join_line("well", "done") == "well done"
