"""Full benchmark execution pipeline, latency profiling, and statistics."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from rag_bench.config import BenchmarkConfig
from rag_bench.data import DatasetSplits, load_beir_dataset
from rag_bench.hardware import collect_manifest
from rag_bench.latency import LatencyTracker, measure_batch_throughput
from rag_bench.metrics import evaluate_run
from rag_bench.retrievers.base import format_doc_text
from rag_bench.retrievers.bm25 import BM25Retriever, tokenize
from rag_bench.retrievers.dense_exact import DenseExactRetriever
from rag_bench.retrievers.dense_hnsw import DenseHNSWRetriever
from rag_bench.retrievers.hybrid_rrf import rrf_fuse
from rag_bench.retrievers.hybrid_score_fusion import score_fuse
from rag_bench.retrievers.reranker import CrossEncoderReranker
from rag_bench.stats import adjust_comparisons, compare_systems
from rag_bench.tuning import tune_rrf_k, tune_score_fusion_alpha, tune_weighted_rrf


class BenchmarkPipeline:
    """Orchestrates end-to-end benchmark execution, tuning, profiling, and reporting."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_dataset(self, dataset_name: str) -> Path:
        """Execute full benchmark pipeline on a single dataset."""
        run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{dataset_name}_{run_timestamp}"
        output_dir = self.results_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n========================================================")
        print(f"STARTING BENCHMARK RUN: {run_id}")
        print(f"Dataset: {dataset_name} | Seed: {self.config.seed}")
        print("========================================================\n")

        # 1. Ingestion
        dataset = load_beir_dataset(
            dataset_name=dataset_name,
            data_dir=self.config.data_dir,
            allow_fiqa=self.config.allow_fiqa,
        )
        print(f"Corpus loaded: {len(dataset.corpus)} documents")
        print(f"Test queries loaded: {len(dataset.test_qrels)} queries")

        # 2. Build First-Stage Retrievers and Track Build Costs
        cost_metrics: Dict[str, Any] = {}

        # BM25 build
        t0 = time.perf_counter()
        bm25_retriever = BM25Retriever(
            k1=self.config.bm25.k1,
            b=self.config.bm25.b,
        )
        bm25_retriever.index(dataset.corpus)
        bm25_build_time = time.perf_counter() - t0
        bm25_mem_bytes = (
            bm25_retriever.matrix.data.nbytes
            + bm25_retriever.matrix.indices.nbytes
            + bm25_retriever.matrix.indptr.nbytes
            if bm25_retriever.matrix is not None
            else 0
        )
        cost_metrics["bm25_index_build_sec"] = float(bm25_build_time)
        cost_metrics["bm25_memory_mb"] = float(bm25_mem_bytes / (1024 * 1024))
        print(f"BM25 index built in {bm25_build_time:.2f}s ({cost_metrics['bm25_memory_mb']:.2f} MB)")

        # Dense exact build (embedding generation)
        dense_cache_path = Path(self.config.data_dir) / dataset_name / "embeddings_bge_small.npy"
        t0 = time.perf_counter()
        dense_retriever = DenseExactRetriever(
            model_name=self.config.models.dense.name,
            revision=self.config.models.dense.revision,
            batch_size=self.config.models.dense.batch_size,
            cache_path=dense_cache_path,
        )
        dense_retriever.index(dataset.corpus)
        dense_embed_time = time.perf_counter() - t0
        dense_mem_bytes = (
            dense_retriever.embeddings.nbytes
            if dense_retriever.embeddings is not None
            else 0
        )
        cost_metrics["dense_embedding_sec"] = float(dense_embed_time)
        cost_metrics["dense_memory_mb"] = float(dense_mem_bytes / (1024 * 1024))
        print(f"Dense embeddings ready in {dense_embed_time:.2f}s ({cost_metrics['dense_memory_mb']:.2f} MB)", flush=True)

        # HNSW build
        t0 = time.perf_counter()
        hnsw_retriever = DenseHNSWRetriever(
            exact_retriever=dense_retriever,
            m=self.config.hnsw.m,
            ef_construction=self.config.hnsw.ef_construction,
            ef_search=self.config.hnsw.ef_search,
        )
        hnsw_retriever.index(dataset.corpus)
        hnsw_build_time = time.perf_counter() - t0
        cost_metrics["hnsw_index_build_sec"] = float(hnsw_build_time)
        print(f"HNSW index built in {hnsw_build_time:.2f}s", flush=True)

        # Reranker loader
        reranker = CrossEncoderReranker(
            model_name=self.config.models.reranker.name,
            revision=self.config.models.reranker.revision,
            max_length=self.config.models.reranker.max_length,
            batch_size=self.config.models.reranker.batch_size,
        )

        # 3. Phase 5: Hyperparameter Tuning on Train/Dev Splits ONLY
        print("\n--- Tuning Hyperparameters on Train/Dev Splits (Zero Test Leakage) ---")
        tuning_split = "dev" if dataset.dev_qrels else "train"
        tune_qrels = dataset.dev_qrels if dataset.dev_qrels else dataset.train_qrels
        tune_queries = {qid: dataset.queries[qid] for qid in tune_qrels if qid in dataset.queries}

        # First stage train/dev runs for tuning
        bm25_tune_run = bm25_retriever.batch_search(tune_queries, top_k=100)
        dense_tune_run = dense_retriever.batch_search(tune_queries, top_k=100)

        # Tune RRF k
        best_k, rrf_k_sweep = tune_rrf_k(
            bm25_tune_run,
            dense_tune_run,
            tune_qrels,
            k_candidates=self.config.tuning.rrf_k_candidates,
            split_name=tuning_split,
        )
        print(f"Tuned RRF k on {tuning_split}: best_k={best_k} (sweep: {rrf_k_sweep})")

        # Tune Weighted RRF w (dense weight fixed to 1.0)
        best_w, weighted_rrf_sweep = tune_weighted_rrf(
            bm25_tune_run,
            dense_tune_run,
            tune_qrels,
            k=best_k,
            bm25_weights=self.config.tuning.weighted_rrf_bm25_weights,
            split_name=tuning_split,
        )
        print(f"Tuned Weighted RRF w on {tuning_split}: best_w={best_w} (sweep: {weighted_rrf_sweep})")

        # Tune Score Fusion alpha
        best_alpha, score_fusion_sweep = tune_score_fusion_alpha(
            bm25_tune_run,
            dense_tune_run,
            tune_qrels,
            alphas=self.config.tuning.score_fusion_alphas,
            split_name=tuning_split,
        )
        print(f"Tuned Score Fusion alpha on {tuning_split}: best_alpha={best_alpha} (sweep: {score_fusion_sweep})")

        tuning_results = {
            "tuning_split": tuning_split,
            "num_tuning_queries": len(tune_queries),
            "best_rrf_k": best_k,
            "rrf_k_sweep": rrf_k_sweep,
            "best_bm25_weight": best_w,
            "weighted_rrf_sweep": weighted_rrf_sweep,
            "best_score_fusion_alpha": best_alpha,
            "score_fusion_sweep": score_fusion_sweep,
        }

        # 4. Phase 6: Full Experiment Matrix Execution on Test Split
        print("\n--- Executing Experiment Matrix on Test Split ---")
        test_queries = {qid: dataset.queries[qid] for qid in dataset.test_qrels if qid in dataset.queries}
        test_qids = list(test_queries.keys())

        # Base first-stage test runs
        bm25_test_run = bm25_retriever.batch_search(test_queries, top_k=100)
        dense_test_run = dense_retriever.batch_search(test_queries, top_k=100)
        hnsw_test_run = hnsw_retriever.batch_search(test_queries, top_k=100)

        # Fused test runs
        hybrid_rrf_default_run: Dict[str, List[Tuple[str, float]]] = {}
        hybrid_rrf_tuned_run: Dict[str, List[Tuple[str, float]]] = {}
        hybrid_weighted_rrf_run: Dict[str, List[Tuple[str, float]]] = {}
        hybrid_score_fusion_run: Dict[str, List[Tuple[str, float]]] = {}

        for qid in test_qids:
            b_list = bm25_test_run.get(qid, [])
            d_list = dense_test_run.get(qid, [])

            hybrid_rrf_default_run[qid] = rrf_fuse(
                {"bm25": b_list, "dense": d_list},
                k=60,
                weights={"bm25": 1.0, "dense": 1.0},
                top_k=100,
            )
            hybrid_rrf_tuned_run[qid] = rrf_fuse(
                {"bm25": b_list, "dense": d_list},
                k=best_k,
                weights={"bm25": 1.0, "dense": 1.0},
                top_k=100,
            )
            hybrid_weighted_rrf_run[qid] = rrf_fuse(
                {"bm25": b_list, "dense": d_list},
                k=best_k,
                weights={"bm25": best_w, "dense": 1.0},
                top_k=100,
            )
            hybrid_score_fusion_run[qid] = score_fuse(
                b_list,
                d_list,
                alpha=best_alpha,
                top_k=100,
            )

        # Experiment matrix runs dict
        all_runs: Dict[str, Dict[str, List[Tuple[str, float]]]] = {
            "bm25": bm25_test_run,
            "dense_exact": dense_test_run,
            "dense_hnsw": hnsw_test_run,
            "hybrid_rrf_default": hybrid_rrf_default_run,
            "hybrid_rrf_tuned": hybrid_rrf_tuned_run,
            "hybrid_weighted_rrf": hybrid_weighted_rrf_run,
            "hybrid_score_fusion": hybrid_score_fusion_run,
        }

        # Cross-Encoder Reranking on {BM25, Dense, Hybrid RRF tuned} at N in {20, 50, 100}
        rerank_bases = {
            "bm25": bm25_test_run,
            "dense": dense_test_run,
            "hybrid_rrf_tuned": hybrid_rrf_tuned_run,
        }

        # Collect UNIQUE candidate pairs to pre-warm cross-encoder cache via high-throughput vectorization
        all_candidate_pairs: set[Tuple[str, str]] = set()
        for base_name, base_run in rerank_bases.items():
            for qid in test_qids:
                q_text = test_queries[qid]
                for doc_id, _ in base_run.get(qid, [])[:100]:
                    if doc_id in dataset.corpus:
                        all_candidate_pairs.add(
                            (q_text, format_doc_text(dataset.corpus[doc_id]))
                        )

        pairs_list = list(all_candidate_pairs)
        print(f"\nPre-warming cross-encoder cache on {len(pairs_list)} UNIQUE candidate pairs (vectorized batching)...", flush=True)
        t_warm_start = time.perf_counter()
        reranker.warm_cache_pairs(pairs_list)
        print(f"Cross-encoder pre-warming completed in {time.perf_counter() - t_warm_start:.2f}s", flush=True)

        print("\nAssembling Cross-Encoder Reranked Runs across N in {20, 50, 100}...", flush=True)
        for base_name, base_run in rerank_bases.items():
            for n in self.config.rerank_candidate_counts:
                system_key = f"{base_name}+rerank_N{n}"
                t_rerank_start = time.perf_counter()
                reranked_run: Dict[str, List[Tuple[str, float]]] = {}
                for qid in test_qids:
                    reranked_run[qid] = reranker.rerank(
                        query=test_queries[qid],
                        candidates=base_run.get(qid, []),
                        corpus=dataset.corpus,
                        top_n=n,
                        top_k=100,
                    )
                elapsed_rerank = time.perf_counter() - t_rerank_start
                cost_metrics[f"rerank_sec_{system_key}"] = float(elapsed_rerank)
                all_runs[system_key] = reranked_run

        # 5. Evaluate Metrics for All Systems
        summary_metrics: Dict[str, Dict[str, float]] = {}
        per_query_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

        for sys_name, run_data in all_runs.items():
            mean_m, per_q = evaluate_run(dataset.test_qrels, run_data)
            summary_metrics[sys_name] = mean_m
            per_query_metrics[sys_name] = per_q
            print(f"System: {sys_name:<28} | nDCG@10: {mean_m['ndcg@10']:.4f} | Recall@100: {mean_m['recall@100']:.4f}", flush=True)

        # 6. Online Latency Profiling (Protocol: 20 warmups discarded, then per-stage timings)
        print("\n--- Measuring Online Stage Latencies (Protocol: 20 warmups discarded) ---", flush=True)
        latency_tracker = LatencyTracker()
        warmup_count = min(self.config.latency_warmup_queries, len(test_qids) // 2)

        # Warmup
        for qid in test_qids[:warmup_count]:
            q_text = test_queries[qid]
            tokenize(q_text)
            b_res = bm25_retriever.search(q_text, top_k=100)
            dense_retriever.encode_queries([q_text])
            hnsw_retriever.search(q_text, top_k=100)
            fused = rrf_fuse({"bm25": b_res, "dense": []}, k=60)
            reranker.rerank(q_text, fused, dataset.corpus, top_n=20)

        # Measured online single-query passes (profile representative test query sample)
        measured_qids = test_qids[warmup_count:warmup_count + 15]
        for qid in measured_qids:
            q_text = test_queries[qid]

            # 1. Tokenize
            t0 = time.perf_counter()
            tokenize(q_text)
            t_tok = (time.perf_counter() - t0) * 1000.0
            latency_tracker.record("tokenize", t_tok)

            # 2. BM25 Search
            t0 = time.perf_counter()
            bm25_res = bm25_retriever.search(q_text, top_k=100)
            t_bm25 = (time.perf_counter() - t0) * 1000.0
            latency_tracker.record("bm25", t_bm25)

            # 3. Query Embed
            t0 = time.perf_counter()
            dense_retriever.encode_queries([q_text])
            t_emb = (time.perf_counter() - t0) * 1000.0
            latency_tracker.record("query_embed", t_emb)

            # 4. ANN Search (HNSW)
            t0 = time.perf_counter()
            hnsw_retriever.search(q_text, top_k=100)
            t_ann = (time.perf_counter() - t0) * 1000.0
            latency_tracker.record("ann", t_ann)

            # 5. Exact Search
            t0 = time.perf_counter()
            exact_res = dense_retriever.search(q_text, top_k=100)
            t_exact = (time.perf_counter() - t0) * 1000.0
            latency_tracker.record("dense_exact", t_exact)

            # 6. Fusion (RRF tuned)
            t0 = time.perf_counter()
            fused_res = rrf_fuse({"bm25": bm25_res, "dense": exact_res}, k=best_k, top_k=100)
            t_fus = (time.perf_counter() - t0) * 1000.0
            latency_tracker.record("fusion", t_fus)

            # 7. Single-query online cross-encoder inference for N in {20, 50, 100}
            for n in [20, 50, 100]:
                pairs_n = [
                    (q_text, format_doc_text(dataset.corpus[d]))
                    for d, _ in fused_res[:n]
                    if d in dataset.corpus
                ]
                t0 = time.perf_counter()
                if pairs_n:
                    _ = reranker.model.predict(
                        pairs_n,
                        batch_size=len(pairs_n),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )
                t_rr = (time.perf_counter() - t0) * 1000.0
                latency_tracker.record(f"rerank_N{n}", t_rr)

        latency_summary = latency_tracker.summary()

        # Compute per-system end-to-end p50, p95, p99 latency
        system_latencies: Dict[str, Dict[str, float]] = {}
        for sys_name in all_runs:
            # Reconstruct per-query latency distribution for this system
            total_durations: List[float] = []
            for i in range(len(measured_qids)):
                t = 0.0
                if "bm25" in sys_name or "hybrid" in sys_name:
                    t += latency_tracker.stage_timings["bm25"][i]
                if "dense" in sys_name or "hybrid" in sys_name:
                    t += latency_tracker.stage_timings["query_embed"][i]
                    if "hnsw" in sys_name:
                        t += latency_tracker.stage_timings["ann"][i]
                    else:
                        t += latency_tracker.stage_timings["dense_exact"][i]
                if "hybrid" in sys_name:
                    t += latency_tracker.stage_timings["fusion"][i]
                if "rerank" in sys_name:
                    if "N20" in sys_name:
                        t += latency_tracker.stage_timings["rerank_N20"][i]
                    elif "N50" in sys_name:
                        t += latency_tracker.stage_timings["rerank_N50"][i]
                    elif "N100" in sys_name:
                        t += latency_tracker.stage_timings["rerank_N100"][i]
                total_durations.append(t)

            arr = np.array(total_durations, dtype=np.float64)
            system_latencies[sys_name] = {
                "p50_ms": float(np.percentile(arr, 50.0)),
                "p95_ms": float(np.percentile(arr, 95.0)),
                "p99_ms": float(np.percentile(arr, 99.0)),
                "mean_ms": float(np.mean(arr)),
            }

        # Measure batch throughput QPS for BM25 and Dense
        qps_bm25 = measure_batch_throughput(
            lambda: bm25_retriever.batch_search(test_queries, top_k=100),
            len(test_queries),
        )
        qps_dense = measure_batch_throughput(
            lambda: dense_retriever.batch_search(test_queries, top_k=100),
            len(test_queries),
        )

        latency_profiles = {
            "stages": latency_summary,
            "systems": system_latencies,
            "batch_throughput": {
                "bm25": qps_bm25,
                "dense_exact": qps_dense,
            },
        }

        # 7. HNSW efSearch Sweep
        print("\nRunning HNSW efSearch sweep vs exact search...")
        hnsw_sweep = hnsw_retriever.sweep_ef_search(
            queries=test_queries,
            exact_results=dense_test_run,
            qrels=dataset.test_qrels,
            ef_search_values=self.config.hnsw.sweep_ef_search,
            top_k=10,
        )

        # 8. Phase 7: Statistical Significance Testing
        print("\n--- Computing Paired Bootstrap CIs & Permutation Tests ---")
        baselines_to_compare = ["bm25", "dense_exact", "dense+rerank_N100"]
        statistical_comparisons: Dict[str, List[Dict[str, Any]]] = {}

        for base_name in baselines_to_compare:
            if base_name not in all_runs:
                continue
            base_scores = [
                per_query_metrics[base_name][qid]["ndcg@10"]
                for qid in test_qids
            ]

            comp_list: List[Dict[str, Any]] = []
            for sys_name in all_runs:
                if sys_name == base_name:
                    continue
                sys_scores = [
                    per_query_metrics[sys_name][qid]["ndcg@10"]
                    for qid in test_qids
                ]
                comp = compare_systems(
                    sys_scores,
                    base_scores,
                    num_samples=10000,
                    seed=self.config.seed,
                )
                comp["system"] = sys_name
                comp["baseline"] = base_name
                comp_list.append(comp)

            # Apply Holm-Bonferroni correction across the family of comparisons
            adjusted = adjust_comparisons(comp_list)
            statistical_comparisons[f"vs_{base_name}"] = adjusted

        # 9. Query Level Win/Loss/Tie & Stratified Analysis
        print("\n--- Computing Stratified and Failure Analyses ---")
        query_analysis = self._compute_query_analysis(
            dataset=dataset,
            per_query_metrics=per_query_metrics,
            all_runs=all_runs,
            best_k=best_k,
        )

        # 10. Hardware and Run Manifest
        manifest = collect_manifest(
            dataset_name=dataset_name,
            dataset_checksums=dataset.checksums,
            seed=self.config.seed,
            extra_metadata={
                "run_id": run_id,
                "timestamp": run_timestamp,
                "cost_metrics": cost_metrics,
            },
        )

        # 11. Write All Outputs to results/<run_id>/
        with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=2)
        with open(output_dir / "per_query_metrics.json", "w", encoding="utf-8") as f:
            json.dump(per_query_metrics, f, indent=2)
        with open(output_dir / "latency_profiles.json", "w", encoding="utf-8") as f:
            json.dump(latency_profiles, f, indent=2)
        with open(output_dir / "cost_metrics.json", "w", encoding="utf-8") as f:
            json.dump(cost_metrics, f, indent=2)
        with open(output_dir / "hnsw_sweep.json", "w", encoding="utf-8") as f:
            json.dump(hnsw_sweep, f, indent=2)
        with open(output_dir / "tuning_results.json", "w", encoding="utf-8") as f:
            json.dump(tuning_results, f, indent=2)
        with open(output_dir / "statistical_tests.json", "w", encoding="utf-8") as f:
            json.dump(statistical_comparisons, f, indent=2)
        with open(output_dir / "query_analysis.json", "w", encoding="utf-8") as f:
            json.dump(query_analysis, f, indent=2)

        print(f"\nAll benchmark results successfully written to: {output_dir}")
        return output_dir

    def _compute_query_analysis(
        self,
        dataset: DatasetSplits,
        per_query_metrics: Dict[str, Dict[str, Dict[str, float]]],
        all_runs: Dict[str, Dict[str, List[Tuple[str, float]]]],
        best_k: int,
    ) -> Dict[str, Any]:
        """Compute win/loss/tie, stratified breakdowns, and extract 10 real queries."""
        hybrid_sys = "hybrid_rrf_tuned"
        dense_sys = "dense_exact"
        bm25_sys = "bm25"

        test_qids = list(dataset.test_qrels.keys())

        # Win/Loss/Tie tables
        def wlt(sys_a: str, sys_b: str) -> Dict[str, int]:
            wins, losses, ties = 0, 0, 0
            for qid in test_qids:
                sa = per_query_metrics[sys_a][qid]["ndcg@10"]
                sb = per_query_metrics[sys_b][qid]["ndcg@10"]
                diff = sa - sb
                if diff > 1e-4:
                    wins += 1
                elif diff < -1e-4:
                    losses += 1
                else:
                    ties += 1
            return {"wins": wins, "losses": losses, "ties": ties, "total": len(test_qids)}

        wlt_hybrid_vs_dense = wlt(hybrid_sys, dense_sys)
        wlt_hybrid_vs_bm25 = wlt(hybrid_sys, bm25_sys)

        # Stratified analysis by:
        # (a) Query length in tokens
        # Short (<= 6 words), Medium (7-12 words), Long (> 12 words)
        strat_len: Dict[str, Dict[str, Any]] = {
            "short (<=6 words)": {"qids": [], "hybrid_mean": 0.0, "dense_mean": 0.0, "bm25_mean": 0.0},
            "medium (7-12 words)": {"qids": [], "hybrid_mean": 0.0, "dense_mean": 0.0, "bm25_mean": 0.0},
            "long (>12 words)": {"qids": [], "hybrid_mean": 0.0, "dense_mean": 0.0, "bm25_mean": 0.0},
        }

        for qid in test_qids:
            q_text = dataset.queries.get(qid, "")
            word_count = len(q_text.split())
            if word_count <= 6:
                cat = "short (<=6 words)"
            elif word_count <= 12:
                cat = "medium (7-12 words)"
            else:
                cat = "long (>12 words)"
            strat_len[cat]["qids"].append(qid)

        for cat, data in strat_len.items():
            q_list = data["qids"]
            if q_list:
                data["hybrid_mean"] = float(np.mean([per_query_metrics[hybrid_sys][q]["ndcg@10"] for q in q_list]))
                data["dense_mean"] = float(np.mean([per_query_metrics[dense_sys][q]["ndcg@10"] for q in q_list]))
                data["bm25_mean"] = float(np.mean([per_query_metrics[bm25_sys][q]["ndcg@10"] for q in q_list]))
            data["query_count"] = len(q_list)

        # (b) BM25 vs Dense gap
        # BM25 Advantage (BM25 - Dense > 0.1), Parity ([-0.1, 0.1]), Dense Advantage (BM25 - Dense < -0.1)
        strat_gap: Dict[str, Dict[str, Any]] = {
            "BM25 advantage (> +0.10)": {"qids": [], "hybrid_mean": 0.0, "dense_mean": 0.0, "bm25_mean": 0.0},
            "Parity ([-0.10, +0.10])": {"qids": [], "hybrid_mean": 0.0, "dense_mean": 0.0, "bm25_mean": 0.0},
            "Dense advantage (< -0.10)": {"qids": [], "hybrid_mean": 0.0, "dense_mean": 0.0, "bm25_mean": 0.0},
        }

        for qid in test_qids:
            b_score = per_query_metrics[bm25_sys][qid]["ndcg@10"]
            d_score = per_query_metrics[dense_sys][qid]["ndcg@10"]
            gap = b_score - d_score
            if gap > 0.10:
                cat = "BM25 advantage (> +0.10)"
            elif gap < -0.10:
                cat = "Dense advantage (< -0.10)"
            else:
                cat = "Parity ([-0.10, +0.10])"
            strat_gap[cat]["qids"].append(qid)

        for cat, data in strat_gap.items():
            q_list = data["qids"]
            if q_list:
                data["hybrid_mean"] = float(np.mean([per_query_metrics[hybrid_sys][q]["ndcg@10"] for q in q_list]))
                data["dense_mean"] = float(np.mean([per_query_metrics[dense_sys][q]["ndcg@10"] for q in q_list]))
                data["bm25_mean"] = float(np.mean([per_query_metrics[bm25_sys][q]["ndcg@10"] for q in q_list]))
            data["query_count"] = len(q_list)

        # Real Query Examples: 5 where Hybrid beats Dense, 5 where Hybrid loses to Dense
        diff_pairs: List[Tuple[str, float]] = []
        for qid in test_qids:
            h_s = per_query_metrics[hybrid_sys][qid]["ndcg@10"]
            d_s = per_query_metrics[dense_sys][qid]["ndcg@10"]
            diff_pairs.append((qid, h_s - d_s))

        diff_pairs.sort(key=lambda x: x[1], reverse=True)
        hybrid_wins_5 = diff_pairs[:5]
        hybrid_losses_5 = sorted(diff_pairs[-5:], key=lambda x: x[1])

        def build_example_details(qid: str, diff: float) -> Dict[str, Any]:
            q_text = dataset.queries.get(qid, "")
            # Find a relevant doc with positive qrel
            rel_docs = [
                (doc, rel)
                for doc, rel in dataset.test_qrels.get(qid, {}).items()
                if rel > 0
            ]
            rel_docs.sort(key=lambda x: x[1], reverse=True)
            top_rel_doc = rel_docs[0][0] if rel_docs else "none"
            doc_obj = dataset.corpus.get(top_rel_doc, {})
            snippet = format_doc_text(doc_obj)[:250] + ("..." if len(format_doc_text(doc_obj)) > 250 else "")

            # Get rank of top_rel_doc in each channel
            def get_rank(run_list: Sequence[Tuple[str, float]], target_id: str) -> str:
                for r, (d, _) in enumerate(run_list, start=1):
                    if d == target_id:
                        return str(r)
                return ">100"

            bm25_rank = get_rank(all_runs[bm25_sys].get(qid, []), top_rel_doc)
            dense_rank = get_rank(all_runs[dense_sys].get(qid, []), top_rel_doc)
            hybrid_rank = get_rank(all_runs[hybrid_sys].get(qid, []), top_rel_doc)

            return {
                "query_id": qid,
                "query_text": q_text,
                "ndcg_diff": float(diff),
                "hybrid_ndcg": per_query_metrics[hybrid_sys][qid]["ndcg@10"],
                "dense_ndcg": per_query_metrics[dense_sys][qid]["ndcg@10"],
                "bm25_ndcg": per_query_metrics[bm25_sys][qid]["ndcg@10"],
                "relevant_doc_id": top_rel_doc,
                "relevant_doc_snippet": snippet,
                "bm25_rank": bm25_rank,
                "dense_rank": dense_rank,
                "hybrid_rank": hybrid_rank,
            }

        examples_hybrid_beats_dense = [build_example_details(q, diff) for q, diff in hybrid_wins_5]
        examples_hybrid_loses_dense = [build_example_details(q, diff) for q, diff in hybrid_losses_5]

        return {
            "win_loss_tie": {
                "hybrid_vs_dense": wlt_hybrid_vs_dense,
                "hybrid_vs_bm25": wlt_hybrid_vs_bm25,
            },
            "stratified_by_length": strat_len,
            "stratified_by_gap": strat_gap,
            "examples_hybrid_beats_dense": examples_hybrid_beats_dense,
            "examples_hybrid_loses_dense": examples_hybrid_loses_dense,
        }
