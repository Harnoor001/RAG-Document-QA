"""T1.15: real MiniLM embeddings rank the planted facts where a reader would expect them."""

import pytest

pytestmark = pytest.mark.integration


def test_t1_15_refund_question_ranks_page_12_first(minilm_store):
    store, manifest = minilm_store
    assert manifest.embed_dim == 384
    hits = store.search(manifest.doc_id, "how long do I have to return an item", k=5)
    assert 12 in hits[0][0].pages, [(c.pages, round(s, 3)) for c, s in hits]


def test_t1_15_discount_question_ranks_page_18_in_top_3(minilm_store):
    store, manifest = minilm_store
    hits = store.search(manifest.doc_id, "employee discount", k=3)
    assert any(18 in c.pages for c, _ in hits), [(c.pages, round(s, 3)) for c, s in hits]
