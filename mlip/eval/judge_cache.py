"""Disk cache for LLM-as-a-judge verdicts.

Judge calls hit a paid API, and re-running the same eval during development would
re-bill them. We cache each verdict on disk keyed by a hash of (question, answer),
scoped under the judge model so changing the model invalidates old entries.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mlip.config import settings


class JudgeCache:
    def __init__(self, model: str, *, enabled: bool | None = None, cache_dir: str | None = None):
        self.model = model
        self.enabled = settings.judge_cache_enabled if enabled is None else enabled
        base = Path(cache_dir or settings.judge_cache_dir)
        # Scope by model so a model swap doesn't reuse stale verdicts.
        self.dir = base / model.replace("/", "_")

    @staticmethod
    def _key(question: str, answer: str) -> str:
        blob = json.dumps([question, answer], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, question: str, answer: str) -> Path:
        return self.dir / f"{self._key(question, answer)}.json"

    def get(self, question: str, answer: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self._path(question, answer)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, question: str, answer: str, verdict: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        self._path(question, answer).write_text(json.dumps(verdict), encoding="utf-8")
