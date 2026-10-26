"""Hyperparameter tuning module with strict data leakage guards."""

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from rag_bench.metrics import evaluate_run
from rag_bench.retrievers.hybrid_rrf import rrf_fuse
from rag_bench.retrievers.hybrid_score_fusion import score_fuse


def _assert_no_test_leakage(split_name: str | None, qrels: Mapping[str, Any]) -> None:
    """Strict guard asserting that test data is never passed to tuning code."""
    if split_name is not None and "test" in split_name.lower():
        raise ValueError(
            f"DATA LEAKAGE DETECTED: split_name='{split_name}' contains 'test'. "
            "Hyperparameters must only be tuned on train/dev splits!"
        )

    # Check for metadata flag or attribute if attached
    if getattr(qrels, "split", None) == "test":
        raise ValueError(
            "DATA LEAKAGE DETECTED: qrels object has split='test'. "
            "Hyperparameters must only be tuned on train/dev splits!"
        )


def tune_rrf_k(
    bm25_train_run: Mapping[str, Sequence[Tuple[str, float]]],
    dense_train_run: Mapping[str, Sequence[Tuple[str, float]]],
    train_qrels: Mapping[str, Mapping[str, float]],
    k_candidates: Sequence[int] = (10, 30, 60, 100),
    split_name: str | None = None,
) -> Tuple[int, Dict[int, float]]:
    """Tune RRF k parameter on train/dev qrels using nDCG@10.

    Returns:
        (best_k, sweep_results_dict)
    """
    _assert_no_test_leakage(split_name, train_qrels)

    results: Dict[int, float] = {}
    best_k = 60
    best_score = -1.0

    for k in k_candidates:
        fused_run: Dict[str, List[Tuple[str, float]]] = {}
        for qid in train_qrels:
            bm25_candidates = bm25_train_run.get(qid, [])
            dense_candidates = dense_train_run.get(qid, [])
            fused_run[qid] = rrf_fuse(
                {"bm25": bm25_candidates, "dense": dense_candidates},
                k=k,
                weights={"bm25": 1.0, "dense": 1.0},
                top_k=100,
            )

        mean_metrics, _ = evaluate_run(train_qrels, fused_run)
        score = mean_metrics.get("ndcg@10", 0.0)
        results[k] = score

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, results


def tune_weighted_rrf(
    bm25_train_run: Mapping[str, Sequence[Tuple[str, float]]],
    dense_train_run: Mapping[str, Sequence[Tuple[str, float]]],
    train_qrels: Mapping[str, Mapping[str, float]],
    k: int = 60,
    bm25_weights: Sequence[float] = (0.25, 0.5, 0.75, 1.0, 1.5),
    split_name: str | None = None,
) -> Tuple[float, Dict[float, float]]:
    """Tune BM25 weight w in weighted RRF on train/dev qrels using nDCG@10.

    Dense weight is fixed to 1.0.
    Returns:
        (best_bm25_weight, sweep_results_dict)
    """
    _assert_no_test_leakage(split_name, train_qrels)

    results: Dict[float, float] = {}
    best_w = 1.0
    best_score = -1.0

    for w in bm25_weights:
        fused_run: Dict[str, List[Tuple[str, float]]] = {}
        for qid in train_qrels:
            bm25_candidates = bm25_train_run.get(qid, [])
            dense_candidates = dense_train_run.get(qid, [])
            fused_run[qid] = rrf_fuse(
                {"bm25": bm25_candidates, "dense": dense_candidates},
                k=k,
                weights={"bm25": w, "dense": 1.0},
                top_k=100,
            )

        mean_metrics, _ = evaluate_run(train_qrels, fused_run)
        score = mean_metrics.get("ndcg@10", 0.0)
        results[w] = score

        if score > best_score:
            best_score = score
            best_w = w

    return best_w, results


def tune_score_fusion_alpha(
    bm25_train_run: Mapping[str, Sequence[Tuple[str, float]]],
    dense_train_run: Mapping[str, Sequence[Tuple[str, float]]],
    train_qrels: Mapping[str, Mapping[str, float]],
    alphas: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    split_name: str | None = None,
) -> Tuple[float, Dict[float, float]]:
    """Tune min-max score fusion alpha on train/dev qrels using nDCG@10.

    Returns:
        (best_alpha, sweep_results_dict)
    """
    _assert_no_test_leakage(split_name, train_qrels)

    results: Dict[float, float] = {}
    best_alpha = 0.5
    best_score = -1.0

    for alpha in alphas:
        fused_run: Dict[str, List[Tuple[str, float]]] = {}
        for qid in train_qrels:
            bm25_candidates = bm25_train_run.get(qid, [])
            dense_candidates = dense_train_run.get(qid, [])
            fused_run[qid] = score_fuse(
                bm25_candidates,
                dense_candidates,
                alpha=alpha,
                top_k=100,
            )

        mean_metrics, _ = evaluate_run(train_qrels, fused_run)
        score = mean_metrics.get("ndcg@10", 0.0)
        results[alpha] = score

        if score > best_score:
            best_score = score
            best_alpha = alpha

    return best_alpha, results
