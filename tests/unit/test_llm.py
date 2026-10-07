import json

import httpx
import pytest

from app.qa.llm import FakeLLM, LLMUnavailableError, OllamaLLM


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ollama_sends_chat_request_with_temperature_zero():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "  Answer [C1]. "}})

    llm = OllamaLLM(model="llama3.2:3b", url="http://ollama:11434/", client=client_for(handler))
    assert llm.generate("sys", "usr") == "Answer [C1]."
    assert seen["url"] == "http://ollama:11434/api/chat"
    body = seen["body"]
    assert body["model"] == "llama3.2:3b" and body["stream"] is False
    assert body["options"]["temperature"] == 0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_t2_12_server_down_raises_helpful_error_after_retries():
    attempts = []

    def handler(request):
        attempts.append(1)
        raise httpx.ConnectError("connection refused", request=request)

    llm = OllamaLLM(url="http://localhost:11434", retries=2, client=client_for(handler))
    with pytest.raises(LLMUnavailableError, match="ollama serve"):
        llm.generate("s", "u")
    assert len(attempts) == 3


def test_t2_12_missing_model_raises_pull_hint_without_retrying():
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(404, json={"error": "model 'llama3.2:3b' not found"})

    llm = OllamaLLM(model="llama3.2:3b", client=client_for(handler))
    with pytest.raises(LLMUnavailableError, match="ollama pull llama3.2:3b"):
        llm.generate("s", "u")
    assert len(attempts) == 1


def test_timeout_is_retried_then_reported():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    llm = OllamaLLM(timeout_s=5, retries=1, client=client_for(handler))
    with pytest.raises(LLMUnavailableError, match="did not answer within 5 s"):
        llm.generate("s", "u")


def test_server_error_recovers_on_retry():
    replies = iter([httpx.Response(500), httpx.Response(200, json={"message": {"content": "ok"}})])
    llm = OllamaLLM(retries=2, client=client_for(lambda request: next(replies)))
    assert llm.generate("s", "u") == "ok"


def test_persistent_server_error_is_reported():
    llm = OllamaLLM(retries=1, client=client_for(lambda request: httpx.Response(503)))
    with pytest.raises(LLMUnavailableError, match="kept failing"):
        llm.generate("s", "u")


def test_bad_request_is_reported():
    llm = OllamaLLM(client=client_for(lambda request: httpx.Response(400, text="bad options")))
    with pytest.raises(LLMUnavailableError, match="400"):
        llm.generate("s", "u")


def test_is_available_checks_model_list():
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "llama3.2:3b"}]})

    assert OllamaLLM(model="llama3.2:3b", client=client_for(handler)).is_available()
    assert not OllamaLLM(model="llama3.1:8b", client=client_for(handler)).is_available()

    def down(request):
        raise httpx.ConnectError("refused", request=request)

    assert not OllamaLLM(client=client_for(down)).is_available()


def test_settings_drive_defaults(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("OLLAMA_URL", "http://gpu-box:11434")
    llm = OllamaLLM()
    assert (llm.name, llm.url, llm.timeout_s) == ("llama3.1:8b", "http://gpu-box:11434", 60.0)


def test_fake_llm_modes():
    seq = FakeLLM(["a", "b"])
    assert [seq.generate("s", "u") for _ in range(3)] == ["a", "b", "b"]
    assert FakeLLM(lambda s, u: u.upper()).generate("s", "hi") == "HI"
    single = FakeLLM("same")
    single.generate("s1", "u1")
    assert single.calls == [("s1", "u1")]
