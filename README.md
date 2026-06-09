# ML Inference Platform

> An end-to-end LLM platform that treats **model quality as a CI gate**: no model or
> prompt reaches the serving layer unless it passes an automated, agentic evaluation —
> and once it's live, you can see exactly how it performs.

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Code style" src="https://img.shields.io/badge/lint-ruff-orange">
</p>

This repo connects two loops that every real LLM product needs but few side-projects show together:

1. **An agentic evaluation pipeline (offline / CI)** — built with **LangGraph**, scoring
   answers for **RAGAS faithfulness**, correctness, and **LLM-as-a-judge** helpfulness,
   with an **A/B harness** and a **GitHub Actions gate that blocks merges on regression**.
2. **A serving layer with full observability (online)** — a **FastAPI** gateway in front of
   a **pluggable inference backend** (Ollama locally, **vLLM** on GPU), with **Prometheus +
   Grafana** dashboards for QPS, TTFT, and cache hit rate.

**MLflow** sits between them as the registry/experiment store — the single source of truth
for "which model is the champion."

> 📖 **New here?** Read [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) — a plain-English guide
> to what was built, every tool used, and *why* (Ollama, Anthropic, RAGAS, vLLM, and more).

## The statistical quality gate

The headline of this project is a CI gate that is **statistically honest** instead of a naive
mean-threshold. For each metric it runs a **paired test on the same questions** between the
candidate and the stored champion — a seeded **paired bootstrap CI** for continuous metrics and
**McNemar** for the binary judge metric — then applies a **multiple-comparison correction**
(Benjamini–Hochberg by default) across every metric × category test. A change fails only when a
regression is **statistically significant after correction**, not when it dips into the noise.

It also gates on things a metric threshold can't see:

- **🎯 Judge-drift detection (the strongest check).** A small human-labeled gold set measures
  **Cohen's κ** between the LLM judge and human verdicts. Swap the judge to a weak model and
  *every quality metric still passes* — but κ collapses and the gate fails. Measured here:
  healthy Claude-Haiku judge **κ = 0.92**, swapped 1B judge **κ = −0.08** (threshold 0.6).
  No aggregate threshold can catch a broken judge; this does.
- **Per-category attribution.** It names the exact `metric × category` that regressed
  (e.g. `answer_correctness / unanswerable`), not just "the average dropped."
- **Abstention-aware scoring.** Faithfulness is undefined for "I don't know" answers, so the
  `unanswerable` bucket is gated on correctness only.

Every PR gets a **sticky comment** showing the naive and statistical verdicts side by side.

### What I tested — and what didn't hold (honest)

I set out to show "naive false-alarms on a quality-neutral change while the statistical gate
passes." **Running the experiment, that claim did not hold on this 30-question setup** — and
reporting that is the point:

| Candidate change | faithfulness Δ | correctness Δ | naive (tol 0.03) | statistical |
|---|--:|--:|:--:|:--:|
| retrieval `k` 3→2 | −0.005 | −0.025 | PASS | PASS |
| retrieval `k` 3→1 | +0.073 | −0.026 | PASS | PASS |
| aggressive brevity | ~0 | −0.10 | FAIL | **FAIL (real regression)** |
| no-op re-eval | −0.012 | +0.002 | PASS | PASS |

**Why:** on *paired* data, a mean drop large enough to clear naive's threshold also tends to be
statistically significant — *unless* the change is high-variance, and a high-variance change is a
*real* regression on the cratered questions (so a statistical "pass" there would be the gate
being too lenient, not a false alarm). The honest, demonstrable value of the statistical gate is
therefore: **(1) judge-drift detection, (2) per-category attribution, and (3) stable,
significance-aware verdicts** under run-to-run noise.

### Known limitation

RAGAS `answer_correctness` intermittently returns `NaN` on some verbose candidate answers (its
internal statement-extraction step fails to parse them; not cleanly correlated with length). The
gate degrades gracefully — those questions are dropped from that metric's paired test and the
per-test `n` is shown — but it reduces power. Fixing it (retry/validate RAGAS extraction or add an
embedding-similarity fallback) is future work.

## Architecture

```mermaid
flowchart TB
    subgraph CI["🧪 Offline / CI loop — quality gate"]
        direction TB
        D[Eval dataset] --> SUT[RAG system under test]
        SUT --> G["LangGraph eval pipeline<br/>· RAGAS faithfulness<br/>· answer correctness<br/>· LLM-as-a-judge"]
        G --> AB[A/B vs champion]
        AB --> GATE{Regression?}
        GATE -- yes --> BLOCK[❌ block PR]
        GATE -- no --> PROMOTE[✅ promote in registry]
    end

    subgraph REG["📦 MLflow"]
        MR[(Model registry +<br/>experiment tracking)]
    end

    subgraph SERVE["⚡ Online loop — serving"]
        direction TB
        REQ[Client] --> API[FastAPI gateway]
        API --> BK["Pluggable backend<br/>Ollama · vLLM · OpenAI"]
        API --> PROM[Prometheus] --> GRAF[Grafana<br/>QPS · TTFT · cache hit]
    end

    G -.logs metrics.-> MR
    PROMOTE --> MR
    MR -.serves champion.-> BK
```

## Screenshots

**Quality gate blocking a regression (CI)** — the gate fails the build when a candidate drops below the champion:

![Quality gate](docs/images/quality-gate.png)

**A/B harness** — two prompt variants compared head-to-head through the eval pipeline:

![A/B harness](docs/images/ab-harness.png)

**Serving observability (Grafana)** — QPS, time-to-first-token, latency, and cache hit rate:

![Grafana serving dashboard](docs/images/grafana-serving.png)

**Experiment tracking (MLflow)** — every eval run logged with params, metrics, and artifacts:

![MLflow runs](docs/images/mlflow-runs.png)

## Quickstart

**Prerequisites:** [`uv`](https://docs.astral.sh/uv/), Docker (for MLflow), and
[Ollama](https://ollama.com) with a small model pulled (`ollama pull llama3.2:1b`)
for local generation. Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`
(used by the eval judge).

```bash
# 1. Install (uv manages Python 3.11 + deps; no system changes)
make install

# 2. Bring up MLflow
make up          # -> MLflow UI at http://localhost:5001

# 3. Sanity check the config
uv run python -m mlip info

# 4. Try the RAG system under test (needs Ollama running locally)
uv run python -m mlip rag ask "Why does dropout improve generalization?"
uv run python -m mlip rag retrieve "bias variance tradeoff"

# 5. Run the eval pipeline (needs ANTHROPIC_API_KEY in .env)
uv run python -m mlip eval run --name baseline --prompt-version v2
uv run python -m mlip eval run --name bare --no-rag   # evaluate the bare model (no retrieval)

# 6. A/B two variants and promote the winner as the "champion"
uv run python -m mlip eval ab --a-prompt v1 --b-prompt v2 --promote-winner
uv run python -m mlip eval champion        # show the current quality bar

# ...or quantify what retrieval actually buys you (RAG vs no-RAG, same model)
uv run python -m mlip eval compare-rag

# 7. The quality gate (what CI runs): fails with exit 1 if a candidate
#    regresses below the champion beyond tolerance
uv run python -m mlip eval gate --report reports/<some-run>.json
```

### Serving + observability

A FastAPI gateway fronts the pluggable backend, streams tokens, and serves an
in-process prompt cache. It exposes Prometheus metrics that a provisioned
Grafana dashboard renders.

```bash
make serve            # FastAPI gateway on :8000 (talks to local Ollama)
make monitoring       # Prometheus + Grafana (docker, monitoring profile)
make grafana          # open the "MLIP — Serving" dashboard at :3000

# generate some traffic
curl -s localhost:8000/generate -H 'content-type: application/json' \
  -d '{"prompt":"What is overfitting?"}'
```

Tracked signals: **QPS**, **TTFT** (p50/p95), end-to-end **latency** (p50/p95),
**cache hit rate**, and output tokens/sec. The serving engine is a swappable
backend (`ollama` locally, `vllm` on GPU, `openai`) selected by config.

### The quality gate (CI)

On every pull request, the [`Quality Gate`](.github/workflows/quality-gate.yml)
workflow re-evaluates the champion config under the PR's code and runs the
**statistical gate** described [above](#the-statistical-quality-gate): it fails
(exit 1) on a *significant* regression after multiple-comparison correction, or
on judge drift (κ below threshold), and posts a sticky comment with the naive
and statistical verdicts side by side. Use `--naive` to make the legacy
mean-threshold the binding decision instead. It needs an `ANTHROPIC_API_KEY`
repository secret (RAG generation runs on a local Ollama model in the runner);
without the secret the live gate skips gracefully. To make a failing gate
actually block a merge, enable branch protection on `main` and mark the check
**Required**.

> The CLI is invoked as `python -m mlip` during development. (An installed
> `mlip` console script also exists for wheel installs.)

## Repository layout

| Path | Purpose |
|------|---------|
| [`mlip/serving/`](mlip/serving/) | FastAPI serving gateway + pluggable inference backends |
| [`mlip/rag/`](mlip/rag/) | The RAG question-answering system that is being evaluated |
| [`mlip/eval/`](mlip/eval/) | LangGraph eval pipeline: RAGAS, LLM-judge, A/B harness |
| [`mlip/cli.py`](mlip/cli.py) | The `mlip` command-line control plane |
| [`data/`](data/) | Curated eval dataset + document corpus |
| [`monitoring/`](monitoring/) | Prometheus config + Grafana dashboards |
| [`.github/workflows/`](.github/workflows/) | The CI quality gate |

## Tech stack

`FastAPI` · `LangGraph` · `RAGAS` · `MLflow` · `vLLM` / `Ollama` · `Prometheus` · `Grafana` · `Docker` · `GitHub Actions` · `uv` · `ruff`

## Build status

This project is built in vertical slices — each one is independently runnable.

- [x] **Slice 0** — Scaffold: structure, tooling, MLflow via Docker
- [x] **Slice 1** — RAG system under test + eval dataset
- [x] **Slice 2** — LangGraph eval pipeline (RAGAS + judge) → MLflow
- [x] **Slice 3** — A/B harness + champion tracking
- [x] **Slice 4** — GitHub Actions quality gate
- [x] **Slice 5** — Serving + Prometheus/Grafana observability
- [ ] **Slice 6** — Polish: diagrams, screenshots, real vLLM benchmark

## License

[MIT](LICENSE)
