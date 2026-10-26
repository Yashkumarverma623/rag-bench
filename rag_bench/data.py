"""Dataset loading and checksum tracking for BEIR benchmarks."""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import requests
from beir import util


@dataclass
class DatasetSplits:
    dataset_name: str
    source_url: str
    checksums: Dict[str, str] = field(default_factory=dict)
    corpus: Dict[str, Dict[str, str]] = field(default_factory=dict)
    queries: Dict[str, str] = field(default_factory=dict)
    train_qrels: Dict[str, Dict[str, float]] = field(default_factory=dict)
    dev_qrels: Optional[Dict[str, Dict[str, float]]] = None
    test_qrels: Dict[str, Dict[str, float]] = field(default_factory=dict)


def compute_sha256(file_path: Path | str) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_qrels_file(file_path: Path) -> Dict[str, Dict[str, float]]:
    """Load BEIR qrels TSV file."""
    qrels: Dict[str, Dict[str, float]] = {}
    if not file_path.exists():
        return qrels

    with open(file_path, "r", encoding="utf-8") as f:
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


def load_beir_dataset(
    dataset_name: str,
    data_dir: str | Path = "datasets",
    allow_fiqa: bool = False,
) -> DatasetSplits:
    """Download and load BEIR dataset with file checksum verification."""
    dataset_name = dataset_name.lower().strip()
    if dataset_name == "fiqa" and not allow_fiqa:
        raise ValueError(
            "FiQA dataset is flag-gated and off by default. Pass allow_fiqa=True to enable."
        )

    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_folder = out_dir / dataset_name

    source_url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"

    if not dataset_folder.exists() or not (dataset_folder / "corpus.jsonl").exists():
        print(f"Downloading {dataset_name} from {source_url}...")
        util.download_and_unzip(source_url, str(out_dir))

    # Compute checksums of key raw files
    checksums: Dict[str, str] = {}
    for filename in ["corpus.jsonl", "queries.jsonl", "qrels/train.tsv", "qrels/dev.tsv", "qrels/test.tsv"]:
        p = dataset_folder / filename
        if p.exists():
            checksums[filename] = compute_sha256(p)

    # 1. Load corpus: {doc_id: {"title": ..., "text": ...}}
    corpus: Dict[str, Dict[str, str]] = {}
    with open(dataset_folder / "corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                corpus[str(item["_id"])] = {
                    "title": item.get("title", ""),
                    "text": item.get("text", ""),
                }

    # 2. Load queries: {qid: text}
    queries: Dict[str, str] = {}
    with open(dataset_folder / "queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                queries[str(item["_id"])] = item["text"]

    # 3. Load qrels splits
    train_qrels = load_qrels_file(dataset_folder / "qrels" / "train.tsv")
    dev_qrels = None
    dev_path = dataset_folder / "qrels" / "dev.tsv"
    if dev_path.exists():
        dev_qrels = load_qrels_file(dev_path)
    test_qrels = load_qrels_file(dataset_folder / "qrels" / "test.tsv")

    return DatasetSplits(
        dataset_name=dataset_name,
        source_url=source_url,
        checksums=checksums,
        corpus=corpus,
        queries=queries,
        train_qrels=train_qrels,
        dev_qrels=dev_qrels,
        test_qrels=test_qrels,
    )
