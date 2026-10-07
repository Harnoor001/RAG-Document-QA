import pytest

from app.index.embedder import HashEmbedder
from app.index.store import DocumentStore
from app.ingest.chunker import Chunk
from app.qa.retriever import Retriever


class StubStore:
    def __init__(self, hits):
        self.hits = hits
        self.asked_k = None

    def search(self, doc_id, query, k=5):
        self.asked_k = k
        return self.hits[:k]


def chunk(i, text, pages=(1,)):
    return Chunk(id=f"doc-{i:04d}", text=text, pages=list(pages))


@pytest.fixture(scope="module")
def handbook_store(tmp_path_factory, handbook):
    store = DocumentStore(tmp_path_factory.mktemp("idx"), HashEmbedder())
    return store, store.ingest(handbook).doc_id


def test_t2_1_returns_at_most_top_k_sorted_by_score(handbook_store):
    store, doc_id = handbook_store
    result = Retriever(store, top_k=5).retrieve(doc_id, "employee discount on merchandise")
    assert 0 < len(result.chunks) <= 5
    scores = [c.score for c in result.chunks]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 + 1e-6 for s in scores)
    assert result.top_score == pytest.approx(scores[0])


def test_k_argument_overrides_default(handbook_store):
    store, doc_id = handbook_store
    assert len(Retriever(store, top_k=5).retrieve(doc_id, "parking", k=2).chunks) == 2


def test_t2_2_near_duplicates_are_deduplicated():
    text = "Refunds are accepted within 30 days of purchase as long as the item is unused."
    hits = [
        (chunk(1, text), 0.9),
        (chunk(2, text + " Thanks."), 0.85),  # near-identical overlapping chunk
        (chunk(3, text[:40]), 0.8),  # contained in the first
        (chunk(4, "Employees must park in Lot C behind the loading dock."), 0.5),
    ]
    store = StubStore(hits)
    result = Retriever(store, top_k=5).retrieve("doc", "refund")
    assert [c.chunk.id for c in result.chunks] == ["doc-0001", "doc-0004"]
    assert result.chunks[0].score == 0.9  # the higher score is the one kept
    assert store.asked_k == 10  # over-fetched to make room for dropped duplicates


def test_t2_3_context_is_trimmed_to_budget():
    hits = [(chunk(i, f"chunk {i} " + "word " * 200), 0.9 - i / 10) for i in range(5)]
    result = Retriever(StubStore(hits), top_k=5, max_context_chars=2500).retrieve("doc", "q")
    assert sum(len(c.chunk.text) for c in result.chunks) <= 2500
    assert [c.chunk.id for c in result.chunks] == ["doc-0000", "doc-0001"]  # lowest scores dropped


def test_best_chunk_kept_even_when_over_budget():
    hits = [(chunk(0, "word " * 500), 0.7)]
    result = Retriever(StubStore(hits), top_k=5, max_context_chars=100).retrieve("doc", "q")
    assert len(result.chunks) == 1


def test_no_hits_gives_empty_result():
    result = Retriever(StubStore([]), top_k=5).retrieve("doc", "q")
    assert result.chunks == [] and result.top_score == 0.0


def test_defaults_come_from_settings(monkeypatch):
    monkeypatch.setenv("TOP_K", "3")
    monkeypatch.setenv("MAX_CONTEXT_CHARS", "1234")
    r = Retriever(StubStore([]))
    assert (r.top_k, r.max_context_chars) == (3, 1234)
