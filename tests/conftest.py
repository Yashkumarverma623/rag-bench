import json
import os
from pathlib import Path
from typing import Dict

# Prevent ir_datasets from querying unauthorized directories on sandboxed environments / Windows
os.environ.setdefault("IR_DATASETS_HOME", str(Path(__file__).parent.parent / ".ir_datasets"))

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "tiny_corpus"


@pytest.fixture
def tiny_corpus() -> Dict[str, Dict[str, str]]:
    corpus = {}
    with open(FIXTURES_DIR / "corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                corpus[item["_id"]] = {"title": item["title"], "text": item["text"]}
    return corpus


@pytest.fixture
def tiny_queries() -> Dict[str, str]:
    queries = {}
    with open(FIXTURES_DIR / "queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                queries[item["_id"]] = item["text"]
    return queries


@pytest.fixture
def tiny_qrels() -> Dict[str, Dict[str, float]]:
    qrels: Dict[str, Dict[str, float]] = {}
    with open(FIXTURES_DIR / "qrels.tsv", "r", encoding="utf-8") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, doc_id, score = parts[0], parts[1], float(parts[2])
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][doc_id] = score
    return qrels
