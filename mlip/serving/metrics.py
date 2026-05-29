"""Prometheus metrics for the serving layer.

These power the Grafana dashboard: QPS (rate of requests), TTFT and end-to-end
latency distributions, and cache hit rate (hits / (hits + misses)).
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "mlip_requests_total",
    "Total generation requests.",
    ["backend", "status"],
)

TTFT = Histogram(
    "mlip_ttft_seconds",
    "Time to first token (seconds).",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
)

LATENCY = Histogram(
    "mlip_request_latency_seconds",
    "End-to-end request latency (seconds).",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
)

OUTPUT_TOKENS = Counter(
    "mlip_output_tokens_total",
    "Approximate number of output tokens generated.",
)

CACHE = Counter(
    "mlip_cache_lookups_total",
    "Prompt-cache lookups.",
    ["result"],  # "hit" | "miss"
)
