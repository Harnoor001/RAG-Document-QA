"""FastAPI service: upload PDFs, list/delete them, and ask questions with cited answers.

uvicorn app.api.main:app --port 8000
"""

from __future__ import annotations

import hashlib
import re
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.config import get_settings
from app.index.store import DocumentNotFoundError, DocumentStore, Manifest
from app.ingest.loader import InvalidPDFError
from app.qa.answerer import Answerer
from app.qa.llm import LLM, LLMUnavailableError, OllamaLLM
from app.qa.retriever import Retriever

from .logging import get_logger

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
LOCALHOST_ORIGINS = r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?"


class AskRequest(BaseModel):
    doc_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class CitationOut(BaseModel):
    chunk_id: str
    pages: list[int]
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    confidence: float
    sources: int
    refused: bool
    grounded: bool
    latency_ms: int
    load_ms: int
    answer_ms: int
    model: str


def create_app(store: DocumentStore | None = None, llm: LLM | None = None) -> FastAPI:
    """Build the app. Tests inject a store and LLM; otherwise they are created once at startup."""
    settings = get_settings()
    log = get_logger(settings.log_level)

    def init(app: FastAPI) -> None:
        if getattr(app.state, "answerer", None) is not None:
            return
        app.state.store = store or DocumentStore()
        app.state.llm = llm or OllamaLLM()
        app.state.answerer = Answerer(Retriever(app.state.store), app.state.llm)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init(app)
        started = time.perf_counter()
        await run_in_threadpool(lambda: app.state.store.embedder.dim)  # load the model now
        log.info(
            "startup",
            extra={
                "embed_model": app.state.store.embedder.name,
                "llm_model": app.state.llm.name,
                "load_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        yield

    app = FastAPI(
        title="RAG Document QA",
        version="0.3.0",
        summary="Local PDF question answering with page citations and confidence scores.",
        lifespan=lifespan,
    )
    if store is not None or llm is not None:
        init(app)  # lets tests use transports that skip the lifespan
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=LOCALHOST_ORIGINS,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    max_bytes = settings.max_upload_mb * 1024 * 1024

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        fields = {
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round((time.perf_counter() - started) * 1000),
        }
        fields |= getattr(request.state, "log_fields", {})
        log.info("request", extra=fields)
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.error("unhandled_error", extra={"path": request.url.path}, exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/health")
    def health() -> dict:
        llm_ = app.state.llm
        probe = getattr(llm_, "probe", None)
        up, has_model = probe() if probe else (True, True)
        return {
            "status": "ok" if up and has_model else "degraded",
            "ollama": "up" if up else "down",
            "model": llm_.name,
            "model_available": has_model,
            "embed_model": app.state.store.embedder.name,
            "documents": len(app.state.store.list_docs()),
        }

    @app.post("/documents", status_code=201, responses={200: {"description": "Already indexed"}})
    async def upload(response: Response, file: Annotated[UploadFile, File()]) -> dict:
        data = await _read_limited(file, max_bytes)
        if data is None:
            raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB.")
        if not data.startswith(b"%PDF-"):
            raise HTTPException(400, "Only PDF files are accepted.")

        st: DocumentStore = app.state.store
        existed = st.has(hashlib.sha256(data).hexdigest())
        try:
            manifest = await run_in_threadpool(st.ingest_bytes, data, _safe_name(file.filename))
        except InvalidPDFError as exc:
            raise HTTPException(400, str(exc)) from exc
        if existed:
            response.status_code = 200  # same bytes as an indexed document: nothing re-embedded
        return _manifest_out(manifest)

    @app.get("/documents")
    def list_documents() -> list[dict]:
        return [_manifest_out(m) for m in app.state.store.list_docs()]

    @app.get("/documents/{doc_id}")
    def get_document(doc_id: str) -> dict:
        return _manifest_out(_resolve(app, doc_id))

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str) -> Response:
        app.state.store.delete(_resolve(app, doc_id).doc_id)
        return Response(status_code=204)

    @app.post("/ask", response_model=AskResponse)
    async def ask(body: AskRequest, request: Request) -> AskResponse:
        manifest = _resolve(app, body.doc_id)
        try:
            answer = await run_in_threadpool(
                app.state.answerer.answer, manifest.doc_id, body.question, body.top_k
            )
        except LLMUnavailableError as exc:
            raise HTTPException(503, str(exc)) from exc
        request.state.log_fields = {
            "doc_id": manifest.doc_id[:12],
            "question_chars": len(body.question),
            "confidence": answer.confidence,
            "refused": answer.refused,
            "sources": answer.sources,
            "answer_ms": answer.answer_ms,
        }
        return AskResponse(**asdict(answer))

    return app


def _resolve(app: FastAPI, doc_id: str) -> Manifest:
    st: DocumentStore = app.state.store
    try:
        return st.load(st.resolve_id(doc_id)).manifest
    except DocumentNotFoundError as exc:
        raise HTTPException(404, f"No document with id {doc_id}.") from exc


def _manifest_out(manifest: Manifest) -> dict:
    return asdict(manifest)


def _safe_name(filename: str | None) -> str:
    name = Path(filename or "").name
    name = re.sub(r"[^\w.\- ]", "_", name).strip() or "upload.pdf"
    return name if name.lower().endswith(".pdf") else f"{name}.pdf"


async def _read_limited(file: UploadFile, max_bytes: int) -> bytes | None:
    """Read the upload, giving up (None) as soon as it passes ``max_bytes``."""
    parts, size = [], 0
    while chunk := await file.read(1024 * 1024):
        size += len(chunk)
        if size > max_bytes:
            return None
        parts.append(chunk)
    return b"".join(parts)


app = create_app()
