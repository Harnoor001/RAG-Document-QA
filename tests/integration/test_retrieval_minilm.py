"""T1.15: real MiniLM embeddings rank the planted facts where a reader would expect them."""

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def indexed(tmp_path_factory, handbook):
    pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")
    from app.index.embedder import MiniLMEmbedder
    from app.index.store import DocumentStore

    embedder = MiniLMEmbedder()
    try:
        assert (
            embedder.dim == 384
        )  # loads the weights (downloads them from Hugging Face on first run)
    except Exception as exc:
        pytest.skip(f"cannot load {embedder.name} (no cached weights and no network?): {exc}")
    store = DocumentStore(tmp_path_factory.mktemp("idx"), embedder)
    return store, store.ingest(handbook)


def test_t1_15_refund_question_ranks_page_12_first(indexed):
    store, manifest = indexed
    assert manifest.embed_dim == 384
    hits = store.search(manifest.doc_id, "how long do I have to return an item", k=5)
    assert 12 in hits[0][0].pages, [(c.pages, round(s, 3)) for c, s in hits]


def test_t1_15_discount_question_ranks_page_18_in_top_3(indexed):
    store, manifest = indexed
    hits = store.search(manifest.doc_id, "employee discount", k=3)
    assert any(18 in c.pages for c, _ in hits), [(c.pages, round(s, 3)) for c, s in hits]
