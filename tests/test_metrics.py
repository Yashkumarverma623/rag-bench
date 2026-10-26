import math

import numpy as np
from ranx import Qrels, Run, evaluate

from rag_bench.metrics import (
    evaluate_run,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_hand_calculated_ndcg_linear():
    """Verify linear gain NDCG formula: DCG = sum(rel_i / log2(i+1))."""
    # qrels: doc1 has rel 2, doc2 has rel 1
    qrels = {"d1": 2.0, "d2": 1.0}

    # Perfect ranking: d1, d2
    assert ndcg_at_k(["d1", "d2"], qrels, k=10) == 1.0

    # Reversed ranking: d2, d1
    # DCG = 1.0 / log2(2) + 2.0 / log2(3) = 1.0 + 1.2618595 = 2.2618595
    # IDCG = 2.0 / log2(2) + 1.0 / log2(3) = 2.0 + 0.6309297 = 2.6309297
    # NDCG = 2.2618595 / 2.6309297 = 0.8597187
    expected = (1.0 / math.log2(2) + 2.0 / math.log2(3)) / (
        2.0 / math.log2(2) + 1.0 / math.log2(3)
    )
    actual = ndcg_at_k(["d2", "d1"], qrels, k=10)
    assert abs(actual - expected) < 1e-6


def test_hand_calculated_recall_mrr_precision_hitrate():
    qrels = {"d1": 1.0, "d2": 2.0, "d3": 1.0}
    retrieved = ["d99", "d2", "d88", "d1"]

    # In top 10: retrieved d2 (rank 2) and d1 (rank 4) -> 2 out of 3 relevant
    assert abs(recall_at_k(retrieved, qrels, k=10) - (2.0 / 3.0)) < 1e-6
    # MRR: first relevant is at rank 2 -> 1/2 = 0.5
    assert abs(mrr_at_k(retrieved, qrels, k=10) - 0.5) < 1e-6
    # HitRate: has relevant doc in top 10 -> 1.0
    assert hit_rate_at_k(retrieved, qrels, k=10) == 1.0
    # Precision@10: 2 relevant docs in top 10 -> 2/10 = 0.2
    assert abs(precision_at_k(retrieved, qrels, k=10) - 0.2) < 1e-6

    # Top 1: only d99 (irrelevant)
    assert recall_at_k(retrieved, qrels, k=1) == 0.0
    assert mrr_at_k(retrieved, qrels, k=1) == 0.0
    assert hit_rate_at_k(retrieved, qrels, k=1) == 0.0
    assert precision_at_k(retrieved, qrels, k=1) == 0.0


def test_cross_validation_against_ranx():
    """Cross-validate pure NumPy implementation against ranx library across multiple queries."""
    rng = np.random.default_rng(12345)

    qrels_dict = {}
    run_dict = {}

    for q_idx in range(50):
        qid = f"q_{q_idx}"
        # Graded qrels 0, 1, 2
        num_qrels = rng.integers(3, 15)
        doc_ids = [f"d_{rng.integers(0, 100)}" for _ in range(num_qrels)]
        qrels_dict[qid] = {
            doc: float(rel) for doc, rel in zip(doc_ids, rng.choice([0, 1, 2], size=len(doc_ids)))
        }

        # Run with 30 retrieved docs
        run_docs = [f"d_{rng.integers(0, 100)}" for _ in range(30)]
        run_scores = sorted(rng.uniform(0.1, 1.0, size=len(run_docs)), reverse=True)
        # ranx run format: {qid: {doc_id: score}}
        run_dict[qid] = {doc: float(score) for doc, score in zip(run_docs, run_scores)}

    # Evaluate using ranx
    ranx_qrels = Qrels(qrels_dict)
    ranx_run = Run(run_dict)
    metrics_to_eval = ["ndcg@10", "recall@10", "recall@100", "mrr@10", "hit_rate@10", "precision@10"]
    ranx_res = evaluate(ranx_qrels, ranx_run, metrics_to_eval, return_mean=False)

    # Evaluate using our pure Python/NumPy evaluate_run
    our_run_format = {
        qid: sorted(docs.items(), key=lambda x: x[1], reverse=True)
        for qid, docs in run_dict.items()
    }
    _, our_per_query = evaluate_run(qrels_dict, our_run_format)

    # Compare query by query in the exact order ranx stores them
    query_ids = list(ranx_qrels.keys())
    for m_idx, m_name in enumerate(metrics_to_eval):
        ranx_vals = ranx_res[m_name]
        for i, qid in enumerate(query_ids):
            our_val = our_per_query[qid][m_name]
            ranx_val = float(ranx_vals[i])
            diff = abs(our_val - ranx_val)
            assert diff < 1e-6, f"Mismatch on {m_name} for {qid}: our={our_val}, ranx={ranx_val}"
