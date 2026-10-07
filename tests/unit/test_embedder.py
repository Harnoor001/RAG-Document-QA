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
    # checked statically: on Python 3.11 isinstance() would evaluate .dim and load the model
    assert all(hasattr(MiniLMEmbedder, attr) for attr in ("dim", "embed"))
    assert MiniLMEmbedder().name


def test_get_embedder_picks_by_name():
    assert isinstance(get_embedder("hash"), HashEmbedder)
    lazy = get_embedder("sentence-transformers/all-MiniLM-L6-v2")
    assert isinstance(lazy, MiniLMEmbedder)
    assert lazy._model is None  # nothing loaded until the first embed


class OldModel:
    """sentence-transformers before get_embedding_dimension existed."""

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        return np.ones((len(texts), 4), dtype=np.float64) * 3


class NewModel(OldModel):
    def get_embedding_dimension(self):
        return 4

    def get_sentence_embedding_dimension(self):
        raise AssertionError("deprecated name must not be called when the new one exists")


@pytest.mark.parametrize("model_cls", [OldModel, NewModel])
def test_minilm_wraps_model_output(monkeypatch, model_cls):
    class FakeModel(model_cls):
        pass

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


MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture
def hf_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return tmp_path


def fake_snapshot(cache, files=("modules.json", "config.json")):
    snap = cache / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    for name in files:
        (snap / name).write_text("{}")


def test_offline_mode_is_set_when_model_is_cached(hf_cache):
    import os

    from app.index.embedder import prefer_offline

    fake_snapshot(hf_cache)
    assert prefer_offline(MODEL) is True
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_offline_mode_left_alone_when_not_cached(hf_cache):
    import os

    from app.index.embedder import is_model_cached, prefer_offline

    fake_snapshot(hf_cache, files=("config.json",))  # incomplete download
    assert not is_model_cached(MODEL)
    assert prefer_offline(MODEL) is False
    assert "HF_HUB_OFFLINE" not in os.environ


def test_explicit_offline_setting_is_respected(hf_cache, monkeypatch):
    import os

    from app.index.embedder import prefer_offline

    fake_snapshot(hf_cache)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    assert prefer_offline(MODEL) is False
    assert os.environ["HF_HUB_OFFLINE"] == "0"


def test_local_model_folder_counts_as_cached(hf_cache, tmp_path):
    from app.index.embedder import is_model_cached

    assert is_model_cached(str(tmp_path))


def test_cache_dir_follows_hf_home(monkeypatch, tmp_path):
    from app.index.embedder import hf_cache_dir

    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert hf_cache_dir() == tmp_path / "hub"
