"""FastAPI serving gateway.

A thin, observable front door over the pluggable inference backend. It streams
tokens from the backend, records Prometheus metrics (QPS, TTFT, latency, cache
hit rate), and short-circuits identical prompts through an in-process cache.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from mlip.serving.backends import ChatBackend, Message, get_backend
from mlip.serving.cache import ResponseCache
from mlip.serving.metrics import CACHE, LATENCY, OUTPUT_TOKENS, REQUESTS, TTFT


class GenerateRequest(BaseModel):
    prompt: str
    system: str | None = None
    temperature: float = 0.0
    max_tokens: int = 256
    stream: bool = False


def _cache_key(model: str, messages: list[Message], temperature: float) -> str:
    blob = json.dumps([model, messages, temperature], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def create_app(backend: ChatBackend | None = None) -> FastAPI:
    app = FastAPI(title="MLIP Serving Gateway", version="0.1.0")
    backend = backend or get_backend()
    cache = ResponseCache()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "backend": backend.name, "model": backend.model}

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/generate")
    def generate(req: GenerateRequest):
        messages: list[Message] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.prompt})
        key = _cache_key(backend.model, messages, req.temperature)

        cached = cache.get(key)
        if cached is not None:
            CACHE.labels("hit").inc()
            REQUESTS.labels(backend.name, "ok").inc()
            TTFT.observe(0.0)
            LATENCY.observe(0.0)
            if req.stream:
                return StreamingResponse(iter([cached]), media_type="text/plain")
            return {"text": cached, "cached": True, "backend": backend.name, "model": backend.model}

        CACHE.labels("miss").inc()

        def produce() -> Iterator[str]:
            """Stream from the backend while recording TTFT / latency / tokens."""
            start = time.perf_counter()
            first = True
            parts: list[str] = []
            try:
                for piece in backend.stream_chat(
                    messages, temperature=req.temperature, max_tokens=req.max_tokens
                ):
                    if first:
                        TTFT.observe(time.perf_counter() - start)
                        first = False
                    parts.append(piece)
                    yield piece
            except Exception:
                REQUESTS.labels(backend.name, "error").inc()
                raise
            else:
                text = "".join(parts)
                LATENCY.observe(time.perf_counter() - start)
                OUTPUT_TOKENS.inc(len(text.split()))
                REQUESTS.labels(backend.name, "ok").inc()
                cache.set(key, text)

        if req.stream:
            return StreamingResponse(produce(), media_type="text/plain")

        try:
            text = "".join(produce()).strip()
        except Exception as exc:  # noqa: BLE001 - surface backend failure as 502
            raise HTTPException(status_code=502, detail=f"backend error: {exc}") from exc
        return {"text": text, "cached": False, "backend": backend.name, "model": backend.model}

    return app
