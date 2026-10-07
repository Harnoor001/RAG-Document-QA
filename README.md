# RAG Document QA

A fully local PDF question-answering system: answers cite their pages as `[p.N]`, carry a
confidence score, and refuse when the document does not contain the answer. Built with
pdfplumber, sentence-transformers, FAISS and Ollama, in three audit-gated phases.

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Ingestion and indexing: PDF -> page-aware chunks -> FAISS index on disk | done |
| 2 | Retrieval and grounded answering with a local LLM | in review |
| 3 | FastAPI service, UI, evaluation harness, Docker | not started |

## Quickstart

```bash
make install                                   # venv + deps (Python 3.11+)
python -m app.ingest tests/fixtures/handbook.pdf
# handbook.pdf · 42 pages · 165 chunks indexed · 8d9ae4a8c463

ollama serve &                                 # in another terminal
ollama pull llama3.2:3b
python -m app.ask 8d9ae4a8c463 "What's the refund window?"
# Refunds are accepted within 30 days of purchase if the item is unused [p.12].
# confidence 0.xx · 1 source · local LLM (llama3.2:3b) · load <ms> · answer <ms>   (illustrative)
# cited: p.12
```

On Windows without `make`, use `.\run.ps1 install`, `.\run.ps1 test`, `.\run.ps1 test-int`
and `.\run.ps1 lint` (same targets as the Makefile). `app.ask` accepts the 12-character id the
ingest command prints, or the full SHA-256.

The first real ingest downloads `all-MiniLM-L6-v2` (~90 MB) from Hugging Face; after that
everything runs offline. Re-ingesting the same file is a no-op that returns the saved manifest.

| Command | What it runs |
| --- | --- |
| `make test` | unit tests (offline, deterministic `HashEmbedder`), coverage gate 85% on `app/ingest` + `app/index` |
| `make test-int` | integration tests with the real MiniLM model and Ollama (each skips with a reason when unavailable) |
| `make lint` | `ruff check` + `ruff format --check` |

## Configuration

Every setting can be overridden with an environment variable (or a `.env` file).

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHUNK_SIZE` | `800` | max characters per chunk |
| `CHUNK_OVERLAP` | `150` | characters shared by consecutive chunks |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | embedding model; `hash` selects the offline test embedder |
| `DATA_DIR` | `data/indexes` | where indexes are stored |
| `TOP_K` | `5` | chunks retrieved per question |
| `CONFIDENCE_THRESHOLD` | `0.35` | below this top-1 cosine score the system refuses without calling the LLM |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `LLM_MODEL` | `llama3.2:3b` | Ollama model, e.g. `llama3.1:8b` |
| `LLM_TIMEOUT_S` | `60` | per-request timeout (2 retries) |
| `OLLAMA_KEEP_ALIVE` | `30m` | how long Ollama keeps the model loaded after a request |
| `MAX_CONTEXT_CHARS` | `6000` | context budget; lowest-scoring chunks are dropped to fit |

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

## How answering works

`Answer.load_ms` is one-off loading paid by that call (embedding model, FAISS index, and the
Ollama model if it was not resident); `Answer.answer_ms` is retrieval plus generation. Each CLI
run is a fresh process, so it pays the embedding-model load every time; the Phase 3 API loads it
once at startup. When the embedding model is already in the Hugging Face cache, `HF_HUB_OFFLINE=1`
is set automatically so loading it makes no network calls (an explicit `HF_HUB_OFFLINE` wins).

- **Retriever** (`app/qa/retriever.py`): top-k FAISS search, drops near-duplicate chunks (one
  contained in the other, or word Jaccard >= 0.8) keeping the higher score, then trims to
  `MAX_CONTEXT_CHARS`.
- **Confidence gate**: if the best cosine score is below `CONFIDENCE_THRESHOLD`, the answer is
  the fixed refusal and the LLM is never called.
- **Prompt** (`app/qa/prompt.py`): context blocks tagged `[C1]..[Ck]` with page headers inside
  `<context>` tags; rules say to answer only from the context, cite every sentence, refuse with
  an exact phrase, and treat the context as text rather than instructions.
- **LLM** (`app/qa/llm.py`): Ollama `/api/chat`, temperature 0 and a fixed seed, timeout and 2
  retries. A stopped server or a missing model raises `LLMUnavailableError` with the command to
  fix it; the CLI prints it as one line and exits with code 3.
- **Citation validation** (`app/qa/answerer.py`): `[C#]` tags become `[p.12]` or `[p.20-21]`,
  tags outside the retrieved set are dropped, and an answer with no valid citation (or the
  refusal phrase) is returned as a refusal. Confidence is the highest score among the chunks
  the answer actually cites.

## Test fixture

`scripts/make_fixture_pdf.py` deterministically generates a 42-page "Acme Store Employee
Handbook" with planted facts on known pages, listed in `tests/fixtures/planted_facts.json`:
the refund policy on p.12, a warranty sentence split across pp.20-21, a shift-allowance table
on p.27, a near-blank p.33, plus extra facts for the Phase 3 golden set.
