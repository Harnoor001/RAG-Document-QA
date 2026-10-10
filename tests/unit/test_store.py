import json
import shutil

import pytest

from app.config import Settings
from app.index.embedder import HashEmbedder
from app.index.store import DocumentNotFoundError, DocumentStore, file_sha256
from app.ingest.chunker import CHUNKER_VERSION


class CountingEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return super().embed(texts)


@pytest.fixture
def store(tmp_path):
    return DocumentStore(tmp_path / "indexes", CountingEmbedder())


def test_t1_10_ingest_persists_three_files_and_load_round_trips(store, handbook):
    manifest = store.ingest(handbook)
    folder = store.data_dir / manifest.doc_id
    assert sorted(p.name for p in folder.iterdir()) == [
        "chunks.jsonl",
        "index.faiss",
        "manifest.json",
    ]
    saved = json.loads((folder / "manifest.json").read_text())
    assert saved["pages"] == 42 and saved["empty_pages"] == [33]
    assert saved["filename"] == "handbook.pdf"

    in_memory = store.load(manifest.doc_id)
    fresh = DocumentStore(store.data_dir, HashEmbedder()).load(manifest.doc_id)
    assert fresh.chunks == in_memory.chunks
    assert fresh.index.ntotal == in_memory.index.ntotal == manifest.chunks
    assert fresh.manifest == manifest


def test_t1_11_reingest_same_file_is_a_noop(store, handbook):
    first = store.ingest(handbook)
    assert store.embedder.calls == 1
    second = DocumentStore(store.data_dir, store.embedder).ingest(handbook)
    assert second == first
    assert store.embedder.calls == 1


def test_reingest_with_different_settings_rebuilds(store, handbook):
    first = store.ingest(handbook)
    other = DocumentStore(store.data_dir, store.embedder, chunk_size=400, overlap=80)
    second = other.ingest(handbook)
    assert second.doc_id == first.doc_id
    assert second.chunks > first.chunks
    assert store.embedder.calls == 2


def test_index_from_older_chunker_is_rebuilt(store, handbook):
    first = store.ingest(handbook)
    path = store.data_dir / first.doc_id / "manifest.json"
    old = json.loads(path.read_text())
    del old["chunker_version"]  # what a pre-version manifest looks like
    path.write_text(json.dumps(old))
    fresh = DocumentStore(store.data_dir, store.embedder)
    assert fresh.ingest(handbook).chunker_version == CHUNKER_VERSION
    assert store.embedder.calls == 2


def test_t1_12_two_pdfs_list_and_delete(store, handbook, make_pdf):
    other = make_pdf("memo.pdf", ["A short memo about the quarterly stock count procedure."])
    a = store.ingest(handbook)
    b = store.ingest(other)
    assert a.doc_id != b.doc_id
    assert {m.doc_id for m in store.list_docs()} == {a.doc_id, b.doc_id}

    store.delete(b.doc_id)
    assert [m.doc_id for m in store.list_docs()] == [a.doc_id]
    assert not (store.data_dir / b.doc_id).exists()
    with pytest.raises(DocumentNotFoundError):
        store.load(b.doc_id)
    with pytest.raises(DocumentNotFoundError):
        store.delete(b.doc_id)


def test_t1_13_chunk_ids_stable_across_fresh_dirs(tmp_path, handbook):
    a = DocumentStore(tmp_path / "a", HashEmbedder())
    b = DocumentStore(tmp_path / "b", HashEmbedder())
    doc_id = a.ingest(handbook).doc_id
    assert b.ingest(handbook).doc_id == doc_id == file_sha256(handbook)
    ids_a = [c.id for c in a.load(doc_id).chunks]
    ids_b = [c.id for c in b.load(doc_id).chunks]
    assert ids_a == ids_b
    assert ids_a[0] == f"{doc_id[:8]}-0000"


def test_t1_14_env_override_changes_chunk_count(monkeypatch, tmp_path, handbook):
    default = DocumentStore(tmp_path / "default", HashEmbedder()).ingest(handbook)
    monkeypatch.setenv("CHUNK_SIZE", "400")
    monkeypatch.setenv("CHUNK_OVERLAP", "80")
    assert Settings().chunk_size == 400
    smaller = DocumentStore(tmp_path / "small", HashEmbedder()).ingest(handbook)
    assert smaller.chunk_size == 400 and smaller.overlap == 80
    assert smaller.chunks > default.chunks


def test_search_returns_sorted_scored_chunks(store, handbook):
    doc_id = store.ingest(handbook).doc_id
    hits = store.search(doc_id, "Refunds are accepted within 30 days of purchase", k=5)
    assert len(hits) == 5
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 + 1e-6 for s in scores)
    assert 12 in hits[0][0].pages
    assert store.search(doc_id, "anything", k=0) == []


def test_search_on_document_with_no_text(store, make_pdf):
    blank = make_pdf("blank.pdf", ["", ""])
    manifest = store.ingest(blank)
    assert manifest.chunks == 0 and manifest.empty_pages == [1, 2]
    assert store.search(manifest.doc_id, "anything") == []


def test_missing_and_unknown_inputs(store, tmp_path):
    with pytest.raises(FileNotFoundError):
        store.ingest(tmp_path / "missing.pdf")
    with pytest.raises(DocumentNotFoundError):
        store.load("0" * 64)
    assert DocumentStore(tmp_path / "never-created", HashEmbedder()).list_docs() == []


def test_half_deleted_document_is_rebuilt(store, handbook):
    manifest = store.ingest(handbook)
    (store.data_dir / manifest.doc_id / "index.faiss").unlink()
    store.ingest(handbook)
    assert store.embedder.calls == 2
    assert (store.data_dir / manifest.doc_id / "index.faiss").is_file()


def test_store_reads_defaults_from_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "from-env"))
    monkeypatch.setenv("EMBED_MODEL", "hash")
    store = DocumentStore()
    assert store.data_dir == tmp_path / "from-env"
    assert isinstance(store.embedder, HashEmbedder)
    assert (store.chunk_size, store.overlap) == (800, 150)


def test_stray_folders_are_ignored_by_list_docs(store, handbook):
    store.ingest(handbook)
    (store.data_dir / "not-a-doc").mkdir()
    shutil.copy(handbook, store.data_dir / "loose.pdf")
    assert len(store.list_docs()) == 1


def test_resolve_id_accepts_unique_prefix(store, handbook, make_pdf):
    a = store.ingest(handbook).doc_id
    store.ingest(make_pdf("memo.pdf", ["A short memo about the stock count."]))
    assert store.resolve_id(a[:12]) == a
    with pytest.raises(DocumentNotFoundError):
        store.resolve_id("")
    with pytest.raises(DocumentNotFoundError):
        store.resolve_id("zzzz")
