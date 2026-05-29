"""Ollama backend — serves open-weight models locally via Metal on macOS.

This is the default generator (the "model under test") and the local stand-in
for what vLLM serves in production: both speak an OpenAI-style chat API over
open-weight models, so swapping Ollama -> vLLM later is a config change.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx

from mlip.config import settings
from mlip.serving.backends.base import ChatBackend, Message


class OllamaBackend(ChatBackend):
    name = "ollama"

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        super().__init__(model or settings.ollama_model)
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def stream_chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Iterator[str]:
        with httpx.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=120.0,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece
