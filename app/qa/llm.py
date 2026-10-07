"""LLM clients: Ollama for real answers, FakeLLM for tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import httpx

from app.config import get_settings


class LLMUnavailableError(RuntimeError):
    """The local LLM cannot answer: Ollama is not running, or the model is not pulled."""


class LLM(Protocol):
    name: str

    def generate(self, system: str, user: str) -> str: ...


class OllamaLLM:
    def __init__(
        self,
        model: str | None = None,
        url: str | None = None,
        timeout_s: float | None = None,
        retries: int = 2,
        client: httpx.Client | None = None,
        keep_alive: str | None = None,
    ):
        settings = get_settings()
        self.name = model or settings.llm_model
        self.url = (url or settings.ollama_url).rstrip("/")
        self.timeout_s = timeout_s if timeout_s is not None else settings.llm_timeout_s
        self.retries = retries
        self.keep_alive = keep_alive or settings.ollama_keep_alive
        self.last_load_s = 0.0  # time Ollama spent loading the model for the last call
        self._client = client or httpx.Client(timeout=self.timeout_s)

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self.name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "seed": 0},
        }
        last_error: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                response = self._client.post(f"{self.url}/api/chat", json=payload)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_error = exc
                continue
            if response.status_code == 404:
                raise LLMUnavailableError(
                    f"Model '{self.name}' is not available in Ollama. Run: ollama pull {self.name}"
                )
            if response.status_code >= 500:
                last_error = httpx.HTTPStatusError(
                    f"Ollama returned {response.status_code}",
                    request=response.request,
                    response=response,
                )
                continue
            if response.status_code != 200:
                raise LLMUnavailableError(
                    f"Ollama rejected the request ({response.status_code}): {response.text[:200]}"
                )
            data = response.json()
            self.last_load_s = data.get("load_duration", 0) / 1e9  # Ollama reports nanoseconds
            return data["message"]["content"].strip()

        if isinstance(last_error, httpx.TimeoutException):
            raise LLMUnavailableError(
                f"Ollama at {self.url} did not answer within {self.timeout_s:g} s "
                f"(tried {self.retries + 1} times)."
            ) from last_error
        if isinstance(last_error, httpx.HTTPStatusError):
            raise LLMUnavailableError(
                f"Ollama at {self.url} kept failing: {last_error}"
            ) from last_error
        raise LLMUnavailableError(
            f"Cannot reach Ollama at {self.url}. Start it with: ollama serve"
        ) from last_error

    def probe(self) -> tuple[bool, bool]:
        """(Ollama reachable, model pulled), using the cheap /api/tags listing."""
        try:
            response = self._client.get(f"{self.url}/api/tags", timeout=3)
            response.raise_for_status()
            models = response.json().get("models", [])
        except (httpx.HTTPError, ValueError):
            return False, False
        names = {m.get("name", "") for m in models}
        return True, self.name in names or f"{self.name}:latest" in names

    def is_available(self) -> bool:
        """True when Ollama is reachable and the model is pulled."""
        return all(self.probe())


class FakeLLM:
    """Returns canned replies (one string, a list in turn, or a function of the prompt)."""

    name = "fake"

    def __init__(self, responses: str | list[str] | Callable[[str, str], str] = ""):
        self._responses = responses
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if callable(self._responses):
            return self._responses(system, user)
        if isinstance(self._responses, list):
            return self._responses[min(len(self.calls), len(self._responses)) - 1]
        return self._responses
