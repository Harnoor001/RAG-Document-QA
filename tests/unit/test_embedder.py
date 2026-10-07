import numpy as np
import pytest

from app.index.embedder import Embedder, HashEmbedder, MiniLMEmbedder, get_embedder


def test_t1_9_embeddings_are_normalised_float32():
    emb = HashEmbedder()
    vectors = emb.embed(["refund within 30 days", "employee discount", ""])
    assert vectors.dtype == np.float32
    assert vectors.shape == (3, emb.dim)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_hash_embedder_is_deterministic_and_discriminative():
    emb = HashEmbedder()
    a1, a2, b = emb.embed(["refund policy days", "refund policy days", "parking lot location"])
    assert np.array_equal(a1, a2)
    assert float(a1 @ a2) > float(a1 @ b)


def test_embedders_satisfy_protocol():
    assert isinstance(HashEmbedder(), Embedder)
    assert isinstance(MiniLMEmbedder(), Embedder)


def test_get_embedder_picks_by_name():
    assert isinstance(get_embedder("hash"), HashEmbedder)
    lazy = get_embedder("sentence-transformers/all-MiniLM-L6-v2")
    assert isinstance(lazy, MiniLMEmbedder)
    assert lazy._model is None  # nothing loaded until the first embed


def test_minilm_wraps_model_output(monkeypatch):
    class FakeModel:
        def get_sentence_embedding_dimension(self):
            return 4

        def encode(self, texts, **kwargs):
            assert kwargs["normalize_embeddings"] is True
            return np.ones((len(texts), 4), dtype=np.float64) * 3

    emb = MiniLMEmbedder("fake")
    monkeypatch.setattr(emb, "_load", lambda: FakeModel())
    out = emb.embed(["a", "b"])
    assert out.dtype == np.float32 and out.shape == (2, 4)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert emb.embed([]).shape == (0, 4)


@pytest.mark.parametrize("text", ["x", "the the the", "Ünïcödé text"])
def test_hash_embedder_handles_odd_text(text):
    v = HashEmbedder(dim=32).embed([text])
    assert v.shape == (1, 32)
