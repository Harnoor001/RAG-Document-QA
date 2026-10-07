import asyncio
import io

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.main import _safe_name, create_app
from app.index.embedder import HashEmbedder
from app.index.store import DocumentStore
from app.qa.llm import FakeLLM, LLMUnavailableError

REFUND = "Refunds are accepted within 30 days of purchase as long as the item is unused."


class ProbeLLM(FakeLLM):
    def __init__(self, reply="Within 30 days if unused [C1].", up=True, has_model=True):
        super().__init__(reply)
        self.state = (up, has_model)

    def probe(self):
        return self.state


@pytest.fixture
def make_client(monkeypatch, tmp_path):
    # HashEmbedder scores run lower than MiniLM's, so answered-path tests use 0.25
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.25")

    def _make(llm=None):
        store = DocumentStore(tmp_path / "idx", HashEmbedder())
        llm = llm or ProbeLLM()
        app = create_app(store=store, llm=llm)
        return TestClient(app), llm

    return _make


@pytest.fixture
def client(make_client):
    with make_client()[0] as c:
        yield c


@pytest.fixture
def doc_id(client, handbook):
    res = client.post("/documents", files={"file": ("handbook.pdf", handbook.read_bytes())})
    assert res.status_code == 201
    return res.json()["doc_id"]


@pytest.mark.parametrize(
    ("state", "status", "ollama"),
    [
        ((True, True), "ok", "up"),
        ((True, False), "degraded", "up"),
        ((False, False), "degraded", "down"),
    ],
)
def test_t3_1_health_reports_ollama_state(make_client, state, status, ollama):
    client, _ = make_client(ProbeLLM(up=state[0], has_model=state[1]))
    with client:
        body = client.get("/health").json()
    assert body["status"] == status and body["ollama"] == ollama
    assert body["model_available"] is state[1]
    assert body["model"] == "fake" and body["embed_model"] == "hash"


def test_t3_2_upload_then_same_file_again(client, handbook):
    first = client.post("/documents", files={"file": ("handbook.pdf", handbook.read_bytes())})
    assert first.status_code == 201
    body = first.json()
    assert body["pages"] == 42 and body["filename"] == "handbook.pdf" and body["chunks"] > 100
    again = client.post("/documents", files={"file": ("copy.pdf", handbook.read_bytes())})
    assert again.status_code == 200
    assert again.json()["doc_id"] == body["doc_id"]


def test_t3_3_non_pdf_rejected(client):
    res = client.post("/documents", files={"file": ("notes.pdf", b"hello, not a pdf")})
    assert res.status_code == 400 and "PDF" in res.json()["detail"]


def test_t3_3_corrupt_pdf_rejected(client):
    res = client.post("/documents", files={"file": ("x.pdf", b"%PDF-1.4 garbage")})
    assert res.status_code == 400


def test_t3_3_too_large_rejected(monkeypatch, make_client):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    client, _ = make_client()
    big = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 10)
    with client:
        res = client.post("/documents", files={"file": ("big.pdf", io.BytesIO(big))})
    assert res.status_code == 413


def test_t3_4_list_get_delete(client, doc_id):
    listed = client.get("/documents").json()
    assert [d["doc_id"] for d in listed] == [doc_id]
    assert client.get(f"/documents/{doc_id}").json()["empty_pages"] == [33]
    assert client.get(f"/documents/{doc_id[:12]}").status_code == 200  # short id works too
    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.get(f"/documents/{doc_id}").status_code == 404
    assert client.delete(f"/documents/{doc_id}").status_code == 404
    assert client.get("/documents").json() == []


def test_t3_5_ask_returns_full_schema(client, doc_id):
    res = client.post("/ask", json={"doc_id": doc_id, "question": REFUND})
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {
        "answer", "citations", "confidence", "sources", "refused", "grounded",
        "latency_ms", "load_ms", "answer_ms", "model",
    }  # fmt: skip
    assert body["answer"] == "Within 30 days if unused [p.12]."
    assert body["refused"] is False and body["sources"] == 1
    assert set(body["citations"][0]) == {"chunk_id", "pages", "score", "snippet"}
    assert body["citations"][0]["pages"] == [12]
    # the confidence is the real cosine score of the cited chunk, not a rescaled number
    assert body["confidence"] == round(body["citations"][0]["score"], 2)


def test_ask_top_k_is_passed_through(client, doc_id):
    res = client.post("/ask", json={"doc_id": doc_id, "question": REFUND, "top_k": 1})
    assert res.status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "x"},
        {"doc_id": "abc", "question": ""},
        {"doc_id": "abc", "question": "   "},
        {"doc_id": "abc", "question": "x" * 501},
        {"doc_id": "abc", "question": "ok", "top_k": 0},
        {"doc_id": "abc", "question": "ok", "top_k": 11},
    ],
)
def test_t3_6_invalid_ask_is_422(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_t3_6_unknown_doc_is_404(client):
    res = client.post("/ask", json={"doc_id": "f" * 64, "question": "anything"})
    assert res.status_code == 404


def test_t3_7_llm_down_is_503_with_message(make_client, handbook):
    class Down(ProbeLLM):
        def generate(self, system, user):
            raise LLMUnavailableError("Cannot reach Ollama at http://localhost:11434.")

    client, _ = make_client(Down())
    with client:
        doc = client.post("/documents", files={"file": ("h.pdf", handbook.read_bytes())}).json()
        res = client.post("/ask", json={"doc_id": doc["doc_id"], "question": REFUND})
    assert res.status_code == 503
    assert res.json() == {"detail": "Cannot reach Ollama at http://localhost:11434."}


def test_t3_8_below_threshold_refuses_without_llm(make_client, handbook):
    client, llm = make_client()
    with client:
        doc = client.post("/documents", files={"file": ("h.pdf", handbook.read_bytes())}).json()
        body = client.post(
            "/ask", json={"doc_id": doc["doc_id"], "question": "Who won the 2018 FIFA World Cup?"}
        ).json()
    assert body["refused"] is True and body["citations"] == []
    assert llm.calls == []


def test_t3_9_ui_is_served(client):
    res = client.get("/")
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/html")
    assert 'id="upload-form"' in res.text and 'type="file"' in res.text


def test_t3_10_concurrent_asks(make_client, handbook):
    client, llm = make_client()
    with client:
        doc_id = client.post("/documents", files={"file": ("h.pdf", handbook.read_bytes())}).json()[
            "doc_id"
        ]
        app = client.app

    async def fire():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            calls = [
                ac.post("/ask", json={"doc_id": doc_id, "question": REFUND}) for _ in range(10)
            ]
            return await asyncio.gather(*calls)

    responses = asyncio.run(fire())
    assert [r.status_code for r in responses] == [200] * 10
    assert len({r.json()["answer"] for r in responses}) == 1
    assert len(llm.calls) == 10


def test_cors_allows_localhost_only(client):
    ok = client.options(
        "/ask",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:3000"
    bad = client.options(
        "/ask",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in bad.headers


def test_request_logs_are_json_without_document_text(client, doc_id, capsys):
    import json

    capsys.readouterr()
    client.post("/ask", json={"doc_id": doc_id, "question": REFUND})
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    ask_log = next(entry for entry in lines if entry.get("path") == "/ask")
    assert ask_log["status"] == 200 and ask_log["doc_id"] == doc_id[:12]
    assert {"confidence", "refused", "latency_ms", "sources"} <= set(ask_log)
    assert "30 days" not in json.dumps(lines)  # no question, answer or chunk text


def test_unexpected_error_is_json_500(make_client, handbook):
    class Broken(ProbeLLM):
        def generate(self, system, user):
            raise RuntimeError("boom")

    client, _ = make_client(Broken())
    client = TestClient(client.app, raise_server_exceptions=False)
    with client:
        doc = client.post("/documents", files={"file": ("h.pdf", handbook.read_bytes())}).json()
        res = client.post("/ask", json={"doc_id": doc["doc_id"], "question": REFUND})
    assert res.status_code == 500
    assert res.json() == {"detail": "Internal server error."}


def test_safe_name():
    assert _safe_name("../../etc/passwd") == "passwd.pdf"
    assert _safe_name("My Report (v2).pdf") == "My Report _v2_.pdf"
    assert _safe_name(None) == "upload.pdf"


def test_default_app_builds_components_in_lifespan(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBED_MODEL", "hash")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")  # nothing listens here
    with TestClient(create_app()) as client:
        body = client.get("/health").json()
    assert body["ollama"] == "down" and body["documents"] == 0
