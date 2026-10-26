"""Pure Python/NumPy evaluation metrics for Information Retrieval.

Implements linear gain nDCG@k matching trec_eval and ranx, Recall@k,
Recall@100, MRR@k, HitRate@k, and Precision@k with full support for
binary and graded relevance judgements.
"""

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank k using linear gain.

    DCG@k = sum_{i=1}^k (rel_i / log2(i + 1))
    Matching trec_eval and ranx default linear gain formulation.
    """
    rel = np.asarray(relevances[:k], dtype=np.float64)
    if rel.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rel.size + 2))
    return float(np.sum(rel / discounts))


def ndcg_at_k(
    ranked_doc_ids: Sequence[str],
    qrels: Mapping[str, float],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k."""
    # Obtain relevance scores for retrieved docs in order
    retrieved_rels = [qrels.get(doc_id, 0.0) for doc_id in ranked_doc_ids[:k]]
    actual_dcg = dcg_at_k(retrieved_rels, k)

    # Ideal ranking: all positive qrels sorted descending, taking top k
    ideal_rels = sorted([rel for rel in qrels.values() if rel > 0], reverse=True)[:k]
    if not ideal_rels:
        return 0.0
    ideal_dcg = dcg_at_k(ideal_rels, k)

    if ideal_dcg <= 0.0:
        return 0.0
    return float(actual_dcg / ideal_dcg)


def recall_at_k(
    ranked_doc_ids: Sequence[str],
    qrels: Mapping[str, float],
    k: int = 10,
    rel_threshold: float = 0.0,
) -> float:
    """Compute Recall at rank k.

    Proportion of relevant documents (qrel > rel_threshold) retrieved in top k.
    """
    total_relevant = sum(1 for rel in qrels.values() if rel > rel_threshold)
    if total_relevant == 0:
        return 0.0

    retrieved_relevant = sum(
        1 for doc_id in ranked_doc_ids[:k] if qrels.get(doc_id, 0.0) > rel_threshold
    )
    return float(retrieved_relevant / total_relevant)


def mrr_at_k(
    ranked_doc_ids: Sequence[str],
    qrels: Mapping[str, float],
    k: int = 10,
    rel_threshold: float = 0.0,
) -> float:
    """Compute Mean Reciprocal Rank at rank k.

    Reciprocal rank of the first relevant document retrieved within top k.
    """
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if qrels.get(doc_id, 0.0) > rel_threshold:
            return float(1.0 / rank)
    return 0.0


def hit_rate_at_k(
    ranked_doc_ids: Sequence[str],
    qrels: Mapping[str, float],
    k: int = 10,
    rel_threshold: float = 0.0,
) -> float:
    """Compute Hit Rate at rank k.

    1.0 if at least one relevant document is retrieved in top k, else 0.0.
    """
    for doc_id in ranked_doc_ids[:k]:
        if qrels.get(doc_id, 0.0) > rel_threshold:
            return 1.0
    return 0.0


def precision_at_k(
    ranked_doc_ids: Sequence[str],
    qrels: Mapping[str, float],
    k: int = 10,
    rel_threshold: float = 0.0,
) -> float:
    """Compute Precision at rank k.

    Proportion of top-k retrieved documents that are relevant.
    """
    if k <= 0:
        return 0.0
    retrieved_relevant = sum(
        1 for doc_id in ranked_doc_ids[:k] if qrels.get(doc_id, 0.0) > rel_threshold
    )
    return float(retrieved_relevant / k)


def compute_query_metrics(
    ranked_doc_ids: Sequence[str],
    qrels: Mapping[str, float],
) -> Dict[str, float]:
    """Compute all standard benchmark metrics for a single query."""
    return {
        "ndcg@10": ndcg_at_k(ranked_doc_ids, qrels, k=10),
        "recall@10": recall_at_k(ranked_doc_ids, qrels, k=10),
        "recall@100": recall_at_k(ranked_doc_ids, qrels, k=100),
        "mrr@10": mrr_at_k(ranked_doc_ids, qrels, k=10),
        "hit_rate@10": hit_rate_at_k(ranked_doc_ids, qrels, k=10),
        "precision@10": precision_at_k(ranked_doc_ids, qrels, k=10),
    }


def evaluate_run(
    qrels: Mapping[str, Mapping[str, float]],
    run: Mapping[str, Sequence[Any]],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Evaluate full run against qrels.

    Returns:
        (mean_metrics, per_query_metrics)
    """
    per_query: Dict[str, Dict[str, float]] = {}
    metric_sums: Dict[str, float] = {
        "ndcg@10": 0.0,
        "recall@10": 0.0,
        "recall@100": 0.0,
        "mrr@10": 0.0,
        "hit_rate@10": 0.0,
        "precision@10": 0.0,
    }

    # Iterate over queries present in both run and qrels
    query_count = 0
    for qid, query_qrels in qrels.items():
        retrieved = run.get(qid, [])
        ranked_ids: List[str] = []
        if retrieved:
            first = retrieved[0]
            if isinstance(first, (tuple, list)):
                ranked_ids = [str(item[0]) for item in retrieved]
            else:
                ranked_ids = [str(item) for item in retrieved]

        q_metrics = compute_query_metrics(ranked_ids, query_qrels)
        per_query[qid] = q_metrics
        for m, v in q_metrics.items():
            metric_sums[m] += v
        query_count += 1

    if query_count == 0:
        return {m: 0.0 for m in metric_sums}, per_query

    mean_metrics = {m: val / query_count for m, val in metric_sums.items()}
    return mean_metrics, per_query
