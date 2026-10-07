# RAG Document QA

A fully local PDF question-answering system: answers cite their pages as `[p.N]`, carry a
confidence score, and refuse when the document does not contain the answer. Built with
pdfplumber, sentence-transformers, FAISS and Ollama, in three audit-gated phases.

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Ingestion and indexing: PDF -> page-aware chunks -> FAISS index on disk | in review |
| 2 | Retrieval and grounded answering with a local LLM | not started |
| 3 | FastAPI service, UI, evaluation harness, Docker | not started |

## Quickstart (Phase 1)

```bash
make install                                   # venv + deps (Python 3.11+)
make fixture                                   # regenerate tests/fixtures/handbook.pdf
python -m app.ingest tests/fixtures/handbook.pdf
# handbook.pdf · 42 pages · 165 chunks indexed · 8d9ae4a8c463
```

The first real ingest downloads `all-MiniLM-L6-v2` (~90 MB) from Hugging Face; after that
everything runs offline. Re-ingesting the same file is a no-op that returns the saved manifest.

| Command | What it runs |
| --- | --- |
| `make test` | unit tests (offline, deterministic `HashEmbedder`), coverage gate 85% on `app/ingest` + `app/index` |
| `make test-int` | integration tests with the real MiniLM model |
| `make lint` | `ruff check` + `ruff format --check` |

## Configuration

Every setting can be overridden with an environment variable (or a `.env` file).

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHUNK_SIZE` | `800` | max characters per chunk |
| `CHUNK_OVERLAP` | `150` | characters shared by consecutive chunks |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | embedding model; `hash` selects the offline test embedder |
| `DATA_DIR` | `data/indexes` | where indexes are stored |

## How ingestion works

- **Loader** (`app/ingest/loader.py`): pdfplumber, 1-based page numbers, whitespace normalised,
  paragraphs rebuilt from line spacing, line-break hyphens removed. Pages with fewer than 20
  characters are flagged `is_empty` (no OCR). Non-PDF, corrupt and password-protected files
  raise `InvalidPDFError`.
- **Chunker** (`app/ingest/chunker.py`): the pages are read as one continuous text so a sentence
  that runs over a page break stays together. Each chunk ends at the strongest boundary that
  fits (paragraph, then sentence, then word) and never mid-word; the next chunk starts inside the
  last `CHUNK_OVERLAP` characters. Each chunk lists every page it touches, e.g. `[20, 21]`.
- **Store** (`app/index/store.py`): one folder per document named by the SHA-256 of the file,
  holding `index.faiss` (`IndexFlatIP` over L2-normalised vectors, so scores are cosine
  similarity), `chunks.jsonl` and `manifest.json`. Writes go to a temp folder and are renamed
  into place.

## Test fixture

`scripts/make_fixture_pdf.py` deterministically generates a 42-page "Acme Store Employee
Handbook" with planted facts on known pages, listed in `tests/fixtures/planted_facts.json`:
the refund policy on p.12, a warranty sentence split across pp.20-21, a shift-allowance table
on p.27, a near-blank p.33, plus extra facts for the Phase 3 golden set.
