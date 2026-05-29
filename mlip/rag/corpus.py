"""Loading the document corpus and the labelled eval dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
EVAL_PATH = DATA_DIR / "eval.jsonl"


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class EvalExample:
    id: str
    question: str
    ground_truth: str
    source_ids: list[str]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(path: Path = CORPUS_PATH) -> list[Document]:
    return [Document(**row) for row in _read_jsonl(path)]


def load_eval(path: Path = EVAL_PATH) -> list[EvalExample]:
    return [EvalExample(**row) for row in _read_jsonl(path)]
