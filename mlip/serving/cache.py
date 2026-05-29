"""A tiny exact-match prompt cache so repeated requests are served instantly.

This is what produces a meaningful cache-hit-rate metric in the dashboard, and
mirrors how real serving stacks short-circuit identical prompts.
"""

from __future__ import annotations

from collections import OrderedDict


class ResponseCache:
    def __init__(self, maxsize: int = 256) -> None:
        self.maxsize = maxsize
        self._store: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> str | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)  # LRU touch
        return self._store[key]

    def set(self, key: str, value: str) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.maxsize:
            self._store.popitem(last=False)
