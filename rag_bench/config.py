"""Configuration schemas and loaders for rag-bench."""

from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field


class DenseModelConfig(BaseModel):
    name: str = "BAAI/bge-small-en-v1.5"
    revision: str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    batch_size: int = 64


class RerankerModelConfig(BaseModel):
    name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    max_length: int = 512
    batch_size: int = 32


class ModelsConfig(BaseModel):
    dense: DenseModelConfig = Field(default_factory=DenseModelConfig)
    reranker: RerankerModelConfig = Field(default_factory=RerankerModelConfig)


class BM25Config(BaseModel):
    k1: float = 1.2
    b: float = 0.75


class HNSWConfig(BaseModel):
    m: int = 32
    ef_construction: int = 64
    ef_search: int = 64
    sweep_ef_search: List[int] = Field(default_factory=lambda: [10, 20, 40, 64, 128, 256])


class TuningConfig(BaseModel):
    rrf_k_candidates: List[int] = Field(default_factory=lambda: [10, 30, 60, 100])
    weighted_rrf_bm25_weights: List[float] = Field(
        default_factory=lambda: [0.25, 0.5, 0.75, 1.0, 1.5]
    )
    score_fusion_alphas: List[float] = Field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )


class BenchmarkConfig(BaseModel):
    seed: int = 42
    data_dir: str = "datasets"
    results_dir: str = "results"
    datasets: List[str] = Field(default_factory=lambda: ["scifact", "nfcorpus"])
    allow_fiqa: bool = False
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    hnsw: HNSWConfig = Field(default_factory=HNSWConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    rerank_candidate_counts: List[int] = Field(default_factory=lambda: [20, 50, 100])
    latency_warmup_queries: int = 20


def load_config(config_path: str | Path | None = None) -> BenchmarkConfig:
    """Load and parse YAML configuration file."""
    if config_path is None:
        default_p = Path("configs/default.yaml")
        if default_p.exists():
            config_path = default_p
        else:
            return BenchmarkConfig()

    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return BenchmarkConfig.model_validate(data)
