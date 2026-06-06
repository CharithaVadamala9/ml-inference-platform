# ML Inference Platform — What We Built & How It Works

A single, plain-English guide to **what this project does**, **what we built**,
**every tool we used and why**, and **how it all fits together**. No jargon left
unexplained.

---

## 1. What we built (in one sentence)

**We built a platform that answers questions from your own documents, automatically
grades how good those answers are, blocks any change that makes the AI worse, and
serves it with live performance dashboards.**

The headline idea: **model quality is treated like a test in CI** — a model that
gets less faithful literally cannot be merged.

---

## 2. What it does — capability by capability

Each capability below is described as *what you give it* → *what you get back*.

### 1. Answers questions from your own documents (grounded Q&A)
- **What it does:** takes a question, finds the most relevant documents, and answers
  using *only* those documents — and cites which ones it used.
- **You give → you get:** a question → a grounded answer + its sources (or an honest
  *"I don't know"* if the answer isn't in the documents).
- **Why it matters:** it doesn't make things up. The answer is traceable to a source
  you can verify — essential for support, legal, medical, or finance.
- **Try it:** `mlip rag ask "Why does dropout help?"`

### 2. Learns from any documents you give it
- **What it does:** drop your own documents into the knowledge base and it answers
  from them immediately — no retraining.
- **You give → you get:** a set of documents → a Q&A system over them.
- **Why it matters:** the same engine becomes a support bot, an internal company
  assistant, or a domain expert just by swapping the documents.

### 3. Automatically grades answer quality
- **What it does:** runs the AI over a set of test questions (with known correct
  answers) and scores each answer three ways: **faithfulness** (stuck to the
  sources?), **correctness** (matches the truth?), **helpfulness** (a strong AI's
  rating).
- **You give → you get:** a configuration (model/prompt/settings) → a **scorecard**
  of those three numbers.
- **Why it matters:** you *measure* quality with numbers instead of eyeballing a few
  answers and hoping.
- **Try it:** `mlip eval run --name baseline`

### 4. Evaluates the model with OR without retrieval
- **What it does:** grades the full RAG system *or* the bare model on its own — and
  can run both to show the difference.
- **You give → you get:** a model → scores with-RAG vs without-RAG, side by side.
- **Why it matters:** proves, with numbers, **what retrieval actually buys you**, and
  means the platform can evaluate *any* model, not just a RAG pipeline.
- **Try it:** `mlip eval run --no-rag` · `mlip eval compare-rag`

### 5. Compares two versions head-to-head (A/B testing)
- **What it does:** runs two variants (e.g., two prompts) through the same evaluation
  and declares a winner per metric.
- **You give → you get:** two configurations → a side-by-side comparison + the winner.
- **Why it matters:** decide changes with evidence. (Our own A/B found the "stricter"
  prompt was actually *worse* — a regression we'd have shipped on instinct.)
- **Try it:** `mlip eval ab --a-prompt v1 --b-prompt v2`

### 6. Remembers the best version (the "champion")
- **What it does:** stores the best-scoring configuration as the **champion** — the
  quality bar everything else must beat.
- **You give → you get:** a winning run → a saved quality bar.
- **Why it matters:** it gives the system a definition of "good enough to ship."
- **Try it:** `mlip eval promote` · `mlip eval champion`

### 7. Blocks bad versions automatically (the quality gate) ⭐
- **What it does:** on every proposed change it re-grades the model and **fails the
  build if quality dropped** below the champion — automatically, on GitHub.
- **You give → you get:** a proposed change → **PASS** (allowed) or **FAIL** (blocked),
  naming the exact metric that regressed.
- **Why it matters:** **the AI cannot get worse without someone noticing.** Most teams
  test that *code* works; this tests that the *AI is still good.* This is the headline.
- **Try it:** `mlip eval gate --report <a-run>.json`

### 8. Serves the model through an API
- **What it does:** exposes the model behind a web endpoint that streams answers and
  instantly reuses (caches) repeated requests.
- **You give → you get:** a prompt over HTTP → a generated answer.
- **Why it matters:** it's how a real application would actually call the model.
- **Try it:** `mlip serve` then `POST /generate`

### 9. Shows live performance dashboards
- **What it does:** tracks in real time **how busy** (QPS), **how fast** (time to
  first token, latency), and **how efficient** (cache hit rate) the serving is.
- **You give → you get:** live traffic → live graphs on a dashboard.
- **Why it matters:** production observability — see if it's healthy, fast, efficient.

### 10. Records every experiment
- **What it does:** logs every evaluation run's settings, scores, and full report.
- **You give → you get:** any run → a searchable history you can compare over time.
- **Why it matters:** you never lose track of what you tried or how it scored.

---

## 3. The problems it solves (at a glance)

| Problem | What the platform does about it |
|---|---|
| The AI **makes things up** | grounds answers in your documents + scores **faithfulness** |
| Quality **silently gets worse** after a change | the **quality gate** blocks the regression in CI |
| *"Is the new version actually better?"* | **A/B testing** + scorecards decide with data |
| *"Why even use RAG?"* | **compare-rag** measures RAG vs no-RAG |
| *"Is it fast / healthy in production?"* | live **Grafana dashboards** (QPS, TTFT, cache) |
| *"What did we try, and how did it score?"* | **MLflow** experiment history |

---

## 4. What you can run (cheat sheet)

| Command | What it does | What you get back |
|---|---|---|
| `mlip rag ask "..."` | answer a question from the documents | grounded answer + sources |
| `mlip eval run` | grade the model on the test set | a scorecard (3 metrics) |
| `mlip eval run --no-rag` | grade the bare model (no documents) | a scorecard (2 metrics) |
| `mlip eval compare-rag` | RAG vs no-RAG, same model | a comparison table |
| `mlip eval ab` | compare two variants | winner per metric |
| `mlip eval promote` | set the current best as champion | a saved quality bar |
| `mlip eval gate` | check a version against the champion | PASS / FAIL (exit 0/1) |
| `mlip serve` | run the serving API | a live `/generate` endpoint |

---

## 5. The big picture: two "doors" + a logbook

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

- **Door 1** *produces* answers. **Door 2** *judges* answers and *gates* releases.
- **MLflow** *remembers* every run.

---

## 6. The flow, step by step

### Asking a question (Door 1)
1. You ask a question.
2. The question is turned into an **embedding** (numbers representing its meaning).
3. The **retriever** finds the documents closest in meaning (**semantic search**).
4. Those documents + your question go to the **LLM** (via Ollama).
5. The LLM answers **using only those documents** and shows sources — or says
   *"I don't know."*

### Grading the model (Door 2)
1. The system runs the model over a fixed list of test questions (known answers).
2. Each answer is graded: **faithfulness**, **correctness**, **helpfulness**.
3. Scores are compared against the **champion** (best so far).
4. If quality dropped, the **quality gate fails the build** → the change is blocked.
5. Everything is logged to **MLflow**.

---

## 7. Every tool — what it is, what it does here, why we chose it

### 🧠 The model & how it runs

**Ollama** — runs open-source LLMs locally (like "Docker for models"), with an
OpenAI-style API. *Here:* it **runs the model that writes the answers** (the ones we
evaluate and serve). *Why:* free, local, private, GPU-accelerated on Mac, and speaks
the same API as vLLM → swappable later with no rewrite.

**Llama 3.2 (1B)** — the open-weight model Ollama runs (small/fast). *Here:* the
"student" that reads documents and writes answers. *Why:* runs instantly on a laptop
for free while still demonstrating the full pipeline.

**vLLM** *(built for, not running by default)* — a high-throughput GPU serving engine.
*Here:* the **production serving option**; our serving layer has a `vllm` backend ready
to point at a GPU. *Why:* the standard way to serve open models at scale; keeping it
pluggable shows the design scales beyond the laptop.

### 🔎 Retrieval (the "RAG" part)

**sentence-transformers** (`all-MiniLM-L6-v2`) — turns text into **embeddings**
(meaning-vectors). *Here:* powers the **retriever** (semantic search). *Why:* free,
local, CPU-friendly, the recognized standard.

**RAG (Retrieval-Augmented Generation)** *(the technique)* — retrieve relevant docs
first, then answer using them (open-book). *Why:* **grounds** answers and **reduces
hallucination** without retraining.

### 📊 Evaluation (the "grading" part)

**LangGraph** — builds multi-step AI workflows as a graph. *Here:* **orchestrates the
eval pipeline** (`generate → score → aggregate`). *Why:* clean, inspectable
orchestration; an in-demand "agentic" skill.

**RAGAS** — a popular RAG-evaluation metrics library. *Here:* computes **faithfulness**
and **answer correctness**. *Why:* the recognized standard for RAG eval.

**Anthropic Claude (Claude Haiku)** — a frontier LLM via API. *Here, two jobs:*
(1) the **LLM-as-a-judge** rating helpfulness; (2) the **brain RAGAS uses** to compute
faithfulness/correctness. *Why:* the grader should be **stronger than the model it
grades**; Claude is reliable, **Haiku is cheap** (cents per run), and using a different
model avoids "grading its own homework."

### 📒 Tracking

**MLflow** — industry-standard experiment tracking + registry. *Here:* records every
run's settings/scores/report and stores the **champion**. *Why:* the standard; gives a
searchable history and the registry the gate compares against.

### ⚡ Serving & observability

**FastAPI** — a fast, modern Python web framework. *Here:* the **serving gateway**
(`/generate`, `/health`, `/metrics`). *Why:* the de-facto standard for ML serving;
supports streaming (needed for time-to-first-token).

**Prometheus** — collects/stores numeric metrics over time. *Here:* gathers **QPS,
TTFT, latency, cache hit rate** every few seconds. *Why:* the industry standard for
metrics.

**Grafana** — draws metrics as live dashboards. *Here:* the "MLIP — Serving" dashboard.
*Why:* the standard pairing with Prometheus.

### 🧰 Infrastructure, tooling & automation

**Docker / Docker Compose** — containers that run each tool anywhere. *Here:* spins up
MLflow + Prometheus + Grafana with one command. *Why:* reproducibility.

**GitHub Actions** — GitHub's CI/automation. *Here:* runs **lint + tests** and the
**quality gate** on every push/PR. *Why:* turns "model quality" into an enforced,
automatic check.

**uv** — a fast, modern Python package/env manager. *Here:* installs locked
dependencies and runs the project. *Why:* fast + reproducible.

**ruff** — a fast linter + formatter. *Why:* keeps code clean; one tool replaces many.

**pytest** — the standard test framework. *Here:* the 20 offline tests. *Why:* proves
the code is real and reliable.

**Pydantic / pydantic-settings** — typed data + config from `.env`. *Here:* one typed
`Settings` object (keys, model names, URLs). *Why:* clean, validated config in one place.

**Typer + Rich** — CLI framework + pretty terminal output. *Here:* the `mlip` command
and its formatted tables. *Why:* makes the platform easy to drive and demo.

---

## 8. Two modes: with RAG and without

- **With RAG (open-book):** the model is given retrieved documents and answers from
  them. Graded on faithfulness + correctness + helpfulness.
- **Without RAG (closed-book):** the model answers from its **own training memory**, no
  documents. Graded on correctness + helpfulness (faithfulness doesn't apply — no
  sources to be faithful to).

Running both (`mlip eval compare-rag`) shows, with numbers, **what retrieval buys you.**

---

## 9. Honest scope & how it scales

- **Today (demo scale):** ~13 documents searched in memory, a 1B model via Ollama,
  single process — perfect for demonstrating the full pipeline for free.
- **To scale up (no rewrite, by design):** swap the in-memory search for a **vector
  database** (FAISS, Qdrant, pgvector); flip the backend from **Ollama to vLLM** on a
  GPU; add batching + async.

The value isn't a novel invention — it's a **clean, working, end-to-end LLMOps
pipeline**: build, evaluate, gate, serve, and monitor an LLM, with the right patterns
to grow.

---

## 10. Quick glossary

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
