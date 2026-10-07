"""Text -> L2-normalised float32 vectors, so inner product equals cosine similarity."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

_TOKEN = re.compile(r"\w+")


@runtime_checkable
class Embedder(Protocol):
    name: str

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class HashEmbedder:
    """Deterministic, dependency-free embedder for unit tests (hashed bag of words + bigrams)."""

    name = "hash"

    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _TOKEN.findall(text.lower())
            features = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)]
            for feat in features:
                digest = hashlib.blake2b(feat.encode(), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self._dim
                out[row, idx] += 1.0 if digest[4] & 1 else -1.0
            if not features:
                out[row, 0] = 1.0  # keep every vector unit-length, even for empty text
        return _normalise(out)


class MiniLMEmbedder:
    """sentence-transformers model, loaded on first use so cached runs never import torch."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name, device="cpu")
        return self._model

    @property
    def dim(self) -> int:
        model = self._load()
        # sentence-transformers renamed this; the old name now raises a FutureWarning
        getter = getattr(model, "get_embedding_dimension", None)
        if getter is None:
            getter = model.get_sentence_embedding_dimension
        return int(getter())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._load().encode(
            texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True
        )
        return _normalise(vectors)


def get_embedder(model_name: str) -> Embedder:
    """``EMBED_MODEL=hash`` selects the offline test embedder; anything else is a ST model."""
    if model_name == "hash":
        return HashEmbedder()
    return MiniLMEmbedder(model_name)
