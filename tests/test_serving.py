"""Tests for the FastAPI serving gateway (stubbed backend, no Ollama)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mlip.serving.app import create_app
from mlip.serving.backends.base import ChatBackend, Message


class FakeBackend(ChatBackend):
    name = "fake"

    def __init__(self) -> None:
        super().__init__("fake-model")
        self.calls = 0

    def chat(
        self, messages: list[Message], *, temperature: float = 0.0, max_tokens: int = 512
    ) -> str:
        self.calls += 1
        return "hello world"


def _client() -> tuple[TestClient, FakeBackend]:
    backend = FakeBackend()
    return TestClient(create_app(backend=backend)), backend


def test_health():
    client, _ = _client()
    body = client.get("/health").json()
    assert body == {"status": "ok", "backend": "fake", "model": "fake-model"}


def test_generate_and_cache():
    client, backend = _client()
    r1 = client.post("/generate", json={"prompt": "hi"}).json()
    assert r1["text"] == "hello world"
    assert r1["cached"] is False
    assert backend.calls == 1

    # Identical prompt -> served from cache, backend not called again.
    r2 = client.post("/generate", json={"prompt": "hi"}).json()
    assert r2["cached"] is True
    assert backend.calls == 1


def test_metrics_exposed():
    client, _ = _client()
    client.post("/generate", json={"prompt": "metrics please"})
    text = client.get("/metrics").text
    assert "mlip_requests_total" in text
    assert "mlip_ttft_seconds" in text
    assert "mlip_cache_lookups_total" in text
