import pytest

from rag_bench.tuning import (
    tune_rrf_k,
    tune_score_fusion_alpha,
    tune_weighted_rrf,
)


def test_leakage_guard_rejects_test_split(tiny_qrels):
    """Leakage guard: a test that fails if any tuning code reads test qrels."""
    bm25_run = {"q1": [("doc2", 1.0)], "q2": [("doc3", 1.0)], "q3": [("doc5", 1.0)]}
    dense_run = {"q1": [("doc2", 0.9)], "q2": [("doc3", 0.9)], "q3": [("doc5", 0.9)]}

    # 1. Reject if split_name contains 'test'
    with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
        tune_rrf_k(bm25_run, dense_run, tiny_qrels, split_name="test")

    with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
        tune_weighted_rrf(bm25_run, dense_run, tiny_qrels, split_name="test_split")

    with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
        tune_score_fusion_alpha(bm25_run, dense_run, tiny_qrels, split_name="beir_test")

    # 2. Reject if qrels object has split='test'
    class LeakedQrels(dict):
        split = "test"

    leaked = LeakedQrels(tiny_qrels)
    with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
        tune_rrf_k(bm25_run, dense_run, leaked)


def test_tuning_succeeds_on_train_dev(tiny_qrels):
    """Verify tuning executes cleanly and without error on train or dev splits."""
    bm25_run = {"q1": [("doc2", 1.0)], "q2": [("doc3", 1.0)], "q3": [("doc5", 1.0)]}
    dense_run = {"q1": [("doc2", 0.9)], "q2": [("doc3", 0.9)], "q3": [("doc5", 0.9)]}

    best_k, k_res = tune_rrf_k(bm25_run, dense_run, tiny_qrels, split_name="train")
    assert best_k in [10, 30, 60, 100]
    assert len(k_res) == 4

    best_w, w_res = tune_weighted_rrf(bm25_run, dense_run, tiny_qrels, split_name="dev")
    assert best_w in [0.25, 0.5, 0.75, 1.0, 1.5]

    best_alpha, a_res = tune_score_fusion_alpha(bm25_run, dense_run, tiny_qrels, split_name="train")
    assert 0.1 <= best_alpha <= 0.9
