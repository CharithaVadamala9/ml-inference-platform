# How It Works — A Plain-English Guide

This document explains **what we built, every tool we used, what each tool does
in this project, and why we chose it.** No jargon left unexplained.

---

## 1. What we built (in one paragraph)

An **end-to-end LLM platform** with two halves joined by a logbook:

1. A **question-answering system** (RAG) that answers from a trusted set of
   documents instead of making things up.
2. An **evaluation + quality-gate system** that automatically grades those
   answers and **blocks any change that makes quality worse** before it ships.

Plus a **serving layer** (an API + live dashboards) so the model can be used and
monitored in production, and **MLflow** recording every experiment.

The headline idea: **model quality is treated like a test in CI** — a model that
gets less faithful literally cannot be merged.

---

## 2. The big picture: two "doors" + a logbook

```
                ┌────────────────── DOOR 1: ASK ──────────────────┐
   question ──► │  retrieve relevant docs ──► LLM writes answer    │ ──► grounded answer
                └──────────────────────────────────────────────────┘
                                      │ answers
                                      ▼
                ┌──────────────── DOOR 2: GRADE ──────────────────┐
                │  score answers (faithfulness / correctness /     │
                │  helpfulness) ──► compare to "champion" ──►      │ ──► PASS / FAIL
                │  GitHub Actions gate blocks bad versions         │
                └──────────────────────────────────────────────────┘
                                      │ logs every run
                                      ▼
                        MLflow (the experiment logbook)

   Serving layer: a FastAPI "front door" runs the model and reports
   live stats (speed, throughput, cache) to Prometheus + Grafana.
```

- **Door 1** *produces* answers.
- **Door 2** *judges* answers and *gates* releases.
- **MLflow** *remembers* every run.

---

## 3. The flow, step by step

### Asking a question (Door 1)
1. You ask a question.
2. The question is turned into an **embedding** (a list of numbers representing
   its meaning).
3. The **retriever** finds the documents whose embeddings are closest in meaning
   (**semantic search**).
4. Those documents + your question are handed to the **LLM** (via Ollama).
5. The LLM writes an answer **using only those documents**, and shows its sources
   — or says *"I don't know"* if the answer isn't there.

### Grading the model (Door 2)
1. The system runs the model over a fixed list of test questions (with known
   correct answers).
2. Each answer is graded three ways: **faithfulness**, **correctness**,
   **helpfulness**.
3. The scores are compared against the **champion** (the best version so far).
4. If quality dropped, the **quality gate fails the build** → the change is blocked.
5. Everything is logged to **MLflow**.

---

## 4. Every tool — what it is, what it does here, and why we chose it

### 🧠 The model & how it runs

**Ollama**
- **What it is:** a tool that runs open-source LLMs locally on your own computer
  (think "Docker for language models"). It exposes an OpenAI-style API at
  `localhost:11434`.
- **What it does in this project:** it **runs the model that writes the answers**
  — both the answers we evaluate (Door 1) and the answers the serving layer
  returns. It's our "generator engine."
- **Why we chose it:** it's **free, local, and private** (no API bills, nothing
  leaves your machine), it's GPU-accelerated on Mac, and it speaks the **same
  API style as vLLM** — so we can swap to vLLM later with no code rewrite.

**Llama 3.2 (1B)**
- **What it is:** the actual open-weight model that Ollama runs (1 billion
  parameters — small and fast).
- **What it does:** it's the "student" that reads documents and writes answers.
- **Why we chose it:** small enough to run instantly on a laptop for free, while
  still good enough to demonstrate the whole pipeline. (In production you'd swap
  in a larger model served by vLLM.)

**vLLM** *(built for, not running by default)*
- **What it is:** a high-performance engine for serving LLMs to many users fast
  (needs an NVIDIA GPU).
- **What it does in this project:** it's the **production serving option**. Our
  serving layer has a `vllm` backend ready to point at a GPU; switching to it is
  a config change, not a rewrite.
- **Why it's here:** it's the standard way to serve open-weight models at scale,
  and keeping it pluggable shows the design scales beyond the laptop demo.

### 🔎 Retrieval (the "RAG" part)

**sentence-transformers** (model: `all-MiniLM-L6-v2`)
- **What it is:** a library that turns text into **embeddings** (meaning-vectors).
- **What it does:** powers the **retriever** — it converts every document and
  every question into numbers so we can find the closest matches by *meaning*,
  not keywords (**semantic search**).
- **Why we chose it:** it's **free, runs locally on CPU**, is the widely-recognized
  standard for embeddings, and needs no API.

**RAG (Retrieval-Augmented Generation)** *(the technique, not a tool)*
- **What it is:** the pattern of *retrieving relevant documents first, then asking
  the model to answer using them* — like an open-book exam.
- **Why we use it:** it **grounds** answers in trusted sources and **reduces
  hallucination**, without retraining the model. It's why the system can answer
  from *your* documents and say "I don't know" when the answer isn't present.

### 📊 Evaluation (the "grading" part)

**LangGraph**
- **What it is:** a framework for building multi-step AI workflows as a graph
  (a flowchart of steps).
- **What it does:** it **orchestrates the eval pipeline** in order:
  `generate answers → score with RAGAS → score with the judge → aggregate`.
- **Why we chose it:** it makes the pipeline clean, inspectable, and easy to
  extend; "agentic orchestration" is also a highly in-demand skill.

**RAGAS**
- **What it is:** a popular library of metrics for evaluating RAG systems.
- **What it does:** computes two of our grades — **faithfulness** (is the answer
  grounded in the retrieved documents?) and **answer correctness** (does it match
  the known-correct answer?).
- **Why we chose it:** it's the **recognized standard** for RAG evaluation, so it's
  credible on a résumé and battle-tested.

**Anthropic Claude (model: Claude Haiku)**
- **What it is:** a frontier LLM from Anthropic, accessed via API.
- **What it does in this project — two jobs:**
  1. It's the **LLM-as-a-judge** — it reads each answer and rates its overall
     helpfulness 1–5 with a short reason.
  2. It's the **brain RAGAS uses** to compute faithfulness and correctness
     (those metrics need a strong LLM to do the judging).
- **Why we chose it:** the **grader should be more capable than the model being
  graded.** Claude is reliable and good at structured judgments, and **Haiku** is
  cheap (a few cents per eval run). Using a different, stronger model as the judge
  also avoids a model "grading its own homework."

### 📒 Tracking

**MLflow**
- **What it is:** an industry-standard tool for experiment tracking and model
  registry ("git for ML experiments").
- **What it does:** records every eval run — the settings used, the scores, and
  the full report — and stores the **champion** (current best). You browse it all
  in a web UI.
- **Why we chose it:** it's the standard for ML experiment tracking, gives you a
  searchable history of every experiment, and provides the "registry" concept the
  quality gate compares against.

### ⚡ Serving & observability

**FastAPI**
- **What it is:** a modern, fast Python web framework for building APIs.
- **What it does:** it's the **serving gateway** — the front door with endpoints
  `/generate` (get an answer), `/health` (is it alive?), and `/metrics` (stats).
- **Why we chose it:** it's the de-facto standard for serving Python/ML services,
  fast, and supports streaming responses (needed to measure time-to-first-token).

**Prometheus**
- **What it is:** a monitoring system that scrapes and stores numeric metrics
  over time.
- **What it does:** every few seconds it collects the serving stats — **QPS**
  (queries/sec), **TTFT** (time to first token), latency, and cache hit rate.
- **Why we chose it:** it's the industry standard for metrics in production.

**Grafana**
- **What it is:** a dashboard tool that draws metrics as live graphs.
- **What it does:** displays the Prometheus metrics on a provisioned "MLIP —
  Serving" dashboard (QPS, TTFT, latency, cache hit rate, tokens/sec).
- **Why we chose it:** it's the standard pairing with Prometheus and makes the
  serving health visible at a glance.

### 🧰 Infrastructure, tooling & automation

**Docker / Docker Compose**
- **What it is:** containers that package each tool so it runs the same anywhere.
- **What it does:** spins up MLflow, Prometheus, and Grafana with one command.
- **Why we chose it:** reproducibility — anyone can run the infra without manual
  setup.

**GitHub Actions**
- **What it is:** GitHub's built-in automation/CI system.
- **What it does:** on every push/PR it runs **lint + tests**, and the **quality
  gate** that blocks a merge if model quality regresses.
- **Why we chose it:** it's free for public repos and turns "model quality" into an
  enforced, automatic check — the project's headline feature.

**uv**
- **What it is:** a fast, modern Python package & environment manager.
- **What it does:** installs the exact right dependencies (locked for
  reproducibility) and runs the project.
- **Why we chose it:** it's far faster than pip and gives a clean, reproducible
  setup — a strong signal of modern tooling.

**ruff**
- **What it is:** an extremely fast Python linter + formatter.
- **What it does:** keeps the code consistently styled and catches issues.
- **Why we chose it:** speed + it replaces several older tools in one.

**pytest**
- **What it is:** the standard Python testing framework.
- **What it does:** runs the 20 automated tests that prove each part works
  (offline, no API needed).
- **Why we chose it:** it's the standard, and the tests signal the code is real
  and reliable.

**Pydantic / pydantic-settings**
- **What it is:** a library for typed data + loading config from environment/`.env`.
- **What it does:** provides one typed `Settings` object — the single source of
  truth for model names, API keys, and service URLs.
- **Why we chose it:** clean, validated configuration in one place.

**Typer + Rich**
- **What it is:** Typer builds command-line interfaces; Rich makes pretty terminal
  output (tables, colors).
- **What it does:** powers the `mlip` command (`mlip rag ask`, `mlip eval run`,
  `mlip eval gate`, …) with nicely formatted output.
- **Why we chose it:** a clean CLI makes the whole platform easy to drive and demo.

---

## 5. Two modes: with RAG and without

The platform can evaluate the model **two ways**:

- **With RAG (open-book):** the model is given retrieved documents and answers
  from them. Graded on faithfulness + correctness + helpfulness.
- **Without RAG (closed-book):** the model answers from its **own training
  memory**, no documents. Graded on correctness + helpfulness (faithfulness
  doesn't apply — there are no sources to be faithful to).

Running both (`mlip eval compare-rag`) shows, with numbers, **what retrieval
actually buys you** — i.e., *why RAG matters.*

---

## 6. Honest scope & how it scales

- **Today (demo scale):** ~13 documents searched in memory, a 1B model via
  Ollama, single process. Perfect for demonstrating the full pipeline for free.
- **To scale up (no rewrite, by design):**
  - Retrieval → swap the in-memory search for a **vector database** (FAISS, Qdrant,
    pgvector).
  - Serving → flip the backend from **Ollama to vLLM** on a GPU.
  - Throughput → add batching + async.

The value of this project isn't a novel invention — it's a **clean, working,
end-to-end implementation of an LLMOps pipeline**: build, evaluate, gate, serve,
and monitor an LLM, with the right patterns to grow.

---

## 7. Quick glossary

| Term | Plain meaning |
|------|---------------|
| **LLM** | the AI model that generates text (e.g., Llama, Claude) |
| **Inference** | the moment the model *answers* (vs. training, when it learns) |
| **RAG** | retrieve documents first, then answer from them (open-book) |
| **Embedding** | text turned into numbers that capture its meaning |
| **Semantic search** | finding matches by meaning, not exact words |
| **Faithfulness** | did the answer stick to the sources (no hallucination)? |
| **LLM-as-a-judge** | using a strong LLM to grade another model's answers |
| **Champion** | the current best version; the quality bar to beat |
| **Quality gate** | a CI check that blocks a change if quality regresses |
| **TTFT** | time to first token — how fast the answer starts appearing |
| **QPS** | queries per second — how much traffic is being served |
