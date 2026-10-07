import json
import re
from pathlib import Path

import pytest

from app.ingest.chunker import Chunk, chunk_pages
from app.ingest.loader import Page

SIZE, OVERLAP = 800, 150
PLANTED = Path(__file__).parents[1] / "fixtures" / "planted_facts.json"


@pytest.fixture(scope="module")
def chunks(pages):
    return chunk_pages(pages, SIZE, OVERLAP, doc_id="0123456789abcdef")


def test_t1_4_chunks_are_bounded_and_non_empty(chunks):
    assert chunks
    for c in chunks:
        assert c.text.strip()
        assert len(c.text) <= SIZE + 50


def test_t1_5_consecutive_chunks_overlap(chunks):
    # overlap lives inside a run of pages; a sentence-ending page boundary starts afresh
    pairs = [(a, b) for a, b in zip(chunks, chunks[1:], strict=False) if a.pages[-1] == b.pages[0]]
    assert pairs
    for a, b in pairs:
        shared = _longest_suffix_prefix(a.text, b.text)
        assert shared >= 40, (a.id, b.id)
        assert shared <= OVERLAP + 1


def test_t1_6_no_chunk_starts_or_ends_mid_word(pages, chunks):
    text = " ".join(p.text for p in pages if not p.is_empty)
    words = set(re.findall(r"\S+", text))
    for c in chunks:
        tokens = c.text.split()
        assert tokens[0] in words, c.id
        assert tokens[-1] in words, c.id


def test_t1_7_warranty_sentence_chunk_spans_pages_20_and_21(chunks):
    sentence = "collect the original receipt and then submit form WC-2"
    holding = [c for c in chunks if sentence in c.text.replace("\n", " ")]
    assert holding
    assert all(c.pages == [20, 21] for c in holding)


def test_only_the_mid_sentence_page_break_is_crossed(chunks):
    # every fixture page ends a sentence except p.20, which runs into p.21
    assert {tuple(c.pages) for c in chunks if len(c.pages) > 1} == {(20, 21)}


def test_planted_facts_sit_in_chunks_citing_exactly_their_pages(chunks):
    facts = json.loads(PLANTED.read_text())["facts"]
    for fact in facts:
        if fact["kind"] != "text":
            continue
        needle = " ".join(fact["text"].split()[:8])
        holding = [c for c in chunks if needle in " ".join(c.text.split())]
        assert holding, fact["id"]
        pages = [c.pages for c in holding]
        assert all(p == fact["pages"] for p in pages), (fact["id"], pages)


def test_sentence_ending_page_is_a_hard_boundary():
    pages = [Page(1, "Page one ends here.", False), Page(2, "Page two starts.", False)]
    chunks = chunk_pages(pages, 100, 10)
    assert [(c.text, c.pages) for c in chunks] == [
        ("Page one ends here.", [1]),
        ("Page two starts.", [2]),
    ]


def test_t1_8_pages_are_valid_and_ordered(chunks):
    for c in chunks:
        assert c.pages == sorted(set(c.pages))
        assert all(1 <= n <= 42 for n in c.pages)
        assert 33 not in c.pages


def test_chunks_cover_every_non_empty_page(pages, chunks):
    covered = {n for c in chunks for n in c.pages}
    assert covered == {p.number for p in pages if not p.is_empty}


def test_chunk_ids_use_doc_prefix_and_sequence(chunks):
    assert [c.id for c in chunks[:2]] == ["01234567-0000", "01234567-0001"]


def test_start_char_points_into_first_page(pages, chunks):
    for c in chunks:
        page_text = pages[c.pages[0] - 1].text
        first_words = " ".join(c.text.split()[:3])
        assert " ".join(page_text[c.start_char :].split()[:3]) == first_words


def test_page_ending_mid_sentence_joins_with_space():
    pages = [Page(1, "The claim goes to the", False), Page(2, "front desk.", False)]
    (only,) = chunk_pages(pages, 100, 10)
    assert only.text == "The claim goes to the front desk."
    assert only.pages == [1, 2]


def test_empty_input_gives_no_chunks():
    assert chunk_pages([], 100, 10) == []
    assert chunk_pages([Page(1, "", True)], 100, 10) == []


def test_word_longer_than_chunk_is_never_cut():
    long_word = "x" * 120
    pages = [Page(1, f"short words {long_word} more words here", False)]
    chunks = chunk_pages(pages, 50, 10)
    assert any(long_word in c.text for c in chunks)
    for c in chunks:
        assert all(tok == long_word or "x" not in tok for tok in c.text.split())


def test_prefers_paragraph_then_sentence_boundaries():
    para1 = "First paragraph sentence one. " * 5
    para2 = "Second paragraph words go here. " * 5
    pages = [Page(1, para1.strip() + "\n\n" + para2.strip(), False)]
    first = chunk_pages(pages, 200, 20)[0]
    assert first.text == para1.strip()


def test_invalid_sizes_rejected():
    with pytest.raises(ValueError):
        chunk_pages([], 100, 100)
    with pytest.raises(ValueError):
        chunk_pages([], 0, 0)


def test_chunk_round_trips_through_dict():
    c = Chunk(id="a-0001", text="hello", pages=[1, 2], start_char=5)
    assert Chunk.from_dict(c.to_dict()) == c


def _longest_suffix_prefix(a: str, b: str) -> int:
    for n in range(min(len(a), len(b)), 0, -1):
        if a.endswith(b[:n]):
            return n
    return 0
