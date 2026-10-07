"""One folder per document, keyed by the SHA-256 of the PDF bytes:

<data_dir>/<doc_id>/index.faiss    FAISS IndexFlatIP over normalised chunk vectors
<data_dir>/<doc_id>/chunks.jsonl   one Chunk per line, in index order
<data_dir>/<doc_id>/manifest.json  what was ingested and with which settings
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np

from app.config import get_settings
from app.index.embedder import Embedder
from app.ingest.chunker import Chunk, chunk_pages
from app.ingest.loader import load_pdf

INDEX_FILE, CHUNKS_FILE, MANIFEST_FILE = "index.faiss", "chunks.jsonl", "manifest.json"


class DocumentNotFoundError(KeyError):
    pass


@dataclass(frozen=True)
class Manifest:
    doc_id: str
    filename: str
    pages: int
    empty_pages: list[int]
    chunks: int
    embed_model: str
    embed_dim: int
    chunk_size: int
    overlap: int
    created_at: str

    @classmethod
    def from_dict(cls, data: dict) -> Manifest:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__})


@dataclass
class LoadedDocument:
    manifest: Manifest
    chunks: list[Chunk]
    index: faiss.Index


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class DocumentStore:
    def __init__(
        self,
        data_dir: str | Path | None = None,
        embedder: Embedder | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ):
        settings = get_settings()
        self.data_dir = Path(data_dir or settings.data_dir)
        if embedder is None:
            from app.index.embedder import get_embedder

            embedder = get_embedder(settings.embed_model)
        self.embedder = embedder
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.overlap = overlap if overlap is not None else settings.chunk_overlap
        self._cache: dict[str, LoadedDocument] = {}
        self._lock = threading.RLock()  # the API serves requests from a thread pool

    # ---- ingestion ---------------------------------------------------------------------------

    def ingest(self, pdf_path: str | Path) -> Manifest:
        with self._lock:
            return self._ingest(Path(pdf_path))

    def ingest_bytes(self, data: bytes, filename: str) -> Manifest:
        """Ingest an uploaded file; ``filename`` is what the manifest records."""
        workdir = Path(tempfile.mkdtemp(prefix="upload-"))
        try:
            path = workdir / Path(filename).name
            path.write_bytes(data)
            return self.ingest(path)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def has(self, doc_id: str) -> bool:
        return self._read_manifest(doc_id) is not None

    def _ingest(self, pdf_path: Path) -> Manifest:
        if not pdf_path.is_file():
            raise FileNotFoundError(f"No such file: {pdf_path}")
        doc_id = file_sha256(pdf_path)

        existing = self._read_manifest(doc_id)
        if existing is not None and self._matches_settings(existing):
            return existing

        pages = load_pdf(pdf_path)
        chunks = chunk_pages(pages, self.chunk_size, self.overlap, doc_id=doc_id)
        vectors = self.embedder.embed([c.text for c in chunks])
        index = faiss.IndexFlatIP(self.embedder.dim)
        if len(chunks):
            index.add(vectors)

        manifest = Manifest(
            doc_id=doc_id,
            filename=pdf_path.name,
            pages=len(pages),
            empty_pages=[p.number for p in pages if p.is_empty],
            chunks=len(chunks),
            embed_model=self.embedder.name,
            embed_dim=self.embedder.dim,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self._write(manifest, chunks, index)
        self._cache[doc_id] = LoadedDocument(manifest, chunks, index)
        return manifest

    def _matches_settings(self, manifest: Manifest) -> bool:
        return (
            manifest.embed_model == self.embedder.name
            and manifest.chunk_size == self.chunk_size
            and manifest.overlap == self.overlap
            and all(
                (self.data_dir / manifest.doc_id / f).is_file() for f in (INDEX_FILE, CHUNKS_FILE)
            )
        )

    def _write(self, manifest: Manifest, chunks: list[Chunk], index: faiss.Index) -> None:
        """Write into a temp folder and rename, so a crash never leaves a half-written document."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=self.data_dir))
        try:
            faiss.write_index(index, str(tmp / INDEX_FILE))
            with (tmp / CHUNKS_FILE).open("w", encoding="utf-8") as fh:
                for chunk in chunks:
                    fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            (tmp / MANIFEST_FILE).write_text(json.dumps(asdict(manifest), indent=2) + "\n")
            final = self.data_dir / manifest.doc_id
            if final.exists():
                shutil.rmtree(final)
            tmp.rename(final)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)

    # ---- reading -----------------------------------------------------------------------------

    def _read_manifest(self, doc_id: str) -> Manifest | None:
        path = self.data_dir / doc_id / MANIFEST_FILE
        if not path.is_file():
            return None
        return Manifest.from_dict(json.loads(path.read_text()))

    def load(self, doc_id: str) -> LoadedDocument:
        with self._lock:
            return self._load(doc_id)

    def _load(self, doc_id: str) -> LoadedDocument:
        if doc_id in self._cache:
            return self._cache[doc_id]
        manifest = self._read_manifest(doc_id)
        if manifest is None:
            raise DocumentNotFoundError(doc_id)
        folder = self.data_dir / doc_id
        index = faiss.read_index(str(folder / INDEX_FILE))
        with (folder / CHUNKS_FILE).open(encoding="utf-8") as fh:
            chunks = [Chunk.from_dict(json.loads(line)) for line in fh if line.strip()]
        doc = LoadedDocument(manifest, chunks, index)
        self._cache[doc_id] = doc
        return doc

    def resolve_id(self, prefix: str) -> str:
        """Expand a unique doc_id prefix (the CLI prints 12 chars) to the full id."""
        matches = [m.doc_id for m in self.list_docs() if m.doc_id.startswith(prefix)]
        if len(matches) != 1 or not prefix:
            raise DocumentNotFoundError(prefix if not matches else f"{prefix} (ambiguous)")
        return matches[0]

    def list_docs(self) -> list[Manifest]:
        if not self.data_dir.is_dir():
            return []
        found = (self._read_manifest(p.name) for p in sorted(self.data_dir.iterdir()) if p.is_dir())
        return [m for m in found if m is not None]

    def delete(self, doc_id: str) -> None:
        folder = self.data_dir / doc_id
        with self._lock:
            if not (folder / MANIFEST_FILE).is_file():
                raise DocumentNotFoundError(doc_id)
            shutil.rmtree(folder)
            self._cache.pop(doc_id, None)

    def warm_up(self, doc_id: str) -> None:
        """Load the index from disk and the embedding model into memory (no-op once loaded)."""
        self.load(doc_id)
        _ = self.embedder.dim

    def search(self, doc_id: str, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        doc = self.load(doc_id)
        if doc.index.ntotal == 0 or k <= 0:
            return []
        vector = np.asarray(self.embedder.embed([query]), dtype=np.float32)
        scores, ids = doc.index.search(vector, min(k, doc.index.ntotal))
        hits = [(doc.chunks[i], float(s)) for s, i in zip(scores[0], ids[0], strict=True) if i >= 0]
        return sorted(hits, key=lambda hit: hit[1], reverse=True)
