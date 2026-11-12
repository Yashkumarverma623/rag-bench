"""Plot generation for retrieval benchmark analysis."""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def generate_pareto_plot(
    dataset_name: str,
    summary_metrics: Dict[str, Dict[str, float]],
    system_latencies: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Generate Pareto plot: nDCG@10 vs p95 online latency per system."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    systems = list(summary_metrics.keys())
    latencies = [system_latencies[s]["p95_ms"] for s in systems]
    ndcgs = [summary_metrics[s]["ndcg@10"] for s in systems]

    # Color mapping by system family
    colors = []
    for s in systems:
        if "rerank" in s:
            colors.append("#d95f02")  # orange
        elif "hybrid" in s:
            colors.append("#7570b3")  # purple
        elif "dense" in s:
            colors.append("#1b9e77")  # green
        else:
            colors.append("#e7298a")  # pink/bm25

    ax.scatter(latencies, ndcgs, c=colors, s=120, edgecolors="black", alpha=0.9, zorder=5)

    for i, sys_name in enumerate(systems):
        ax.annotate(
            sys_name,
            (latencies[i], ndcgs[i]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            alpha=0.85,
        )

    ax.set_xlabel("Online Latency p95 (ms)", fontsize=11, fontweight="bold")
    ax.set_ylabel("nDCG@10", fontsize=11, fontweight="bold")
    ax.set_title(f"nDCG@10 vs. p95 Online Latency ({dataset_name})", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_hnsw_sweep_plot(
    dataset_name: str,
    hnsw_sweep_data: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate HNSW recall@10 and nDCG loss vs efSearch plot."""
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=200)

    ef_vals = [d["ef_search"] for d in hnsw_sweep_data]
    recalls = [d["ann_recall_at_10_vs_exact"] for d in hnsw_sweep_data]
    ndcg_losses = [d["ndcg_loss_vs_exact"] for d in hnsw_sweep_data]

    ax1.plot(ef_vals, recalls, marker="o", color="#1b9e77", linewidth=2, label="Recall@10 vs Exact")
    ax1.set_xlabel("efSearch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("ANN Recall@10 vs Exact", color="#1b9e77", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#1b9e77")
    ax1.set_ylim(float(min(min(recalls) - 0.05, 0.8)), 1.02)

    ax2 = ax1.twinx()
    ax2.plot(ef_vals, ndcg_losses, marker="s", linestyle="--", color="#d95f02", linewidth=2, label="nDCG@10 Loss")
    ax2.set_ylabel("nDCG@10 Loss vs Exact", color="#d95f02", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#d95f02")

    ax1.set_title(f"HNSW efSearch Sweep ({dataset_name})", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_tuning_sweep_plots(
    dataset_name: str,
    tuning_data: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot RRF k sweep on train/dev split."""
    rrf_k_sweep = tuning_data.get("rrf_k_sweep", {})
    if rrf_k_sweep:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
        ks = sorted([int(k) for k in rrf_k_sweep.keys()])
        scores = [float(rrf_k_sweep.get(k, rrf_k_sweep.get(str(k), 0.0))) for k in ks]
        ax.plot(ks, scores, marker="o", color="#7570b3", linewidth=2)
        ax.set_xlabel("RRF k", fontsize=11, fontweight="bold")
        ax.set_ylabel("nDCG@10", fontsize=11, fontweight="bold")
        ax.set_title(f"RRF k Hyperparameter Tuning ({dataset_name})", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_dir / f"rrf_k_sweep_{dataset_name}.png")
        plt.close()


def generate_reranker_n_sweep_plot(
    dataset_name: str,
    summary_metrics: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Plot Reranker candidate count N sweep across pipelines."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

    ns = [20, 50, 100]
    pipelines = {
        "BM25 + Rerank": [summary_metrics.get(f"bm25+rerank_N{n}", {}).get("ndcg@10", 0.0) for n in ns],
        "Dense + Rerank": [summary_metrics.get(f"dense+rerank_N{n}", {}).get("ndcg@10", 0.0) for n in ns],
        "Hybrid RRF + Rerank": [summary_metrics.get(f"hybrid_rrf_tuned+rerank_N{n}", {}).get("ndcg@10", 0.0) for n in ns],
    }

    colors = ["#e7298a", "#1b9e77", "#7570b3"]
    for (name, vals), color in zip(pipelines.items(), colors):
        ax.plot(ns, vals, marker="o", linewidth=2, label=name, color=color)

    ax.set_xlabel("Candidate Count N", fontsize=11, fontweight="bold")
    ax.set_ylabel("nDCG@10", fontsize=11, fontweight="bold")
    ax.set_title(f"Cross-Encoder Candidate Count N Sweep ({dataset_name})", fontsize=13, fontweight="bold")
    ax.set_xticks(ns)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
