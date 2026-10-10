"""T3.12: the full HTTP flow with the real MiniLM model and Ollama."""

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def test_t3_12_upload_and_ask_over_http(ollama, minilm_store, handbook):
    from app.api.main import create_app

    store, _ = minilm_store
    with TestClient(create_app(store=store, llm=ollama)) as client:
        health = client.get("/health").json()
        assert health["ollama"] == "up" and health["model_available"] is True
        up = client.post("/documents", files={"file": ("handbook.pdf", handbook.read_bytes())})
        assert up.status_code in (200, 201) and up.json()["pages"] == 42
        body = client.post(
            "/ask", json={"doc_id": up.json()["doc_id"], "question": "What's the refund window?"}
        ).json()
    print(f"T3.12 confidence {body['confidence']}: {body['answer']}")
    assert body["refused"] is False, body
    assert "30" in body["answer"] and "[p.12]" in body["answer"], body
