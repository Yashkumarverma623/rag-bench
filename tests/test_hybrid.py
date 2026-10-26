
from rag_bench.retrievers.hybrid_rrf import rrf_fuse
from rag_bench.retrievers.hybrid_score_fusion import score_fuse


def test_rrf_hand_computed_example():
    """Verify RRF hand-computed calculation: score = sum(w / (k + rank))."""
    # Channel 1: d1 (rank 1), d2 (rank 2), d3 (rank 3)
    # Channel 2: d2 (rank 1), d3 (rank 2), d1 (rank 3)
    c1 = [("d1", 10.0), ("d2", 8.0), ("d3", 6.0)]
    c2 = [("d2", 0.95), ("d3", 0.85), ("d1", 0.70)]
    channels = {"c1": c1, "c2": c2}

    # With k=60, equal weights 1.0:
    # d1 score: 1/(60 + 1) + 1/(60 + 3) = 1/61 + 1/63 = 0.01639344 + 0.01587302 = 0.03226646
    # d2 score: 1/(60 + 2) + 1/(60 + 1) = 1/62 + 1/61 = 0.01612903 + 0.01639344 = 0.03252247
    # d3 score: 1/(60 + 3) + 1/(60 + 2) = 1/63 + 1/62 = 0.01587302 + 0.01612903 = 0.03200205
    # Ranking must be: d2 > d1 > d3
    fused = rrf_fuse(channels, k=60)
    assert [doc_id for doc_id, _ in fused] == ["d2", "d1", "d3"]

    expected_d2 = 1.0 / 62.0 + 1.0 / 61.0
    expected_d1 = 1.0 / 61.0 + 1.0 / 63.0
    expected_d3 = 1.0 / 63.0 + 1.0 / 62.0

    scores_map = dict(fused)
    assert abs(scores_map["d2"] - expected_d2) < 1e-6
    assert abs(scores_map["d1"] - expected_d1) < 1e-6
    assert abs(scores_map["d3"] - expected_d3) < 1e-6


def test_rrf_scale_invariance_property():
    """Multiplying input scores by positive constants must NOT alter RRF output rankings."""
    c1 = [("d1", 12.5), ("d2", 8.2), ("d3", 5.1), ("d4", 1.0)]
    c2 = [("d3", 0.99), ("d1", 0.85), ("d4", 0.45), ("d2", 0.10)]
    channels = {"c1": c1, "c2": c2}

    base_fused = rrf_fuse(channels, k=60)

    # Scale channel 1 by 10,000 and channel 2 by 0.0001
    scaled_c1 = [(doc, s * 10000.0) for doc, s in c1]
    scaled_c2 = [(doc, s * 0.0001) for doc, s in c2]
    scaled_channels = {"c1": scaled_c1, "c2": scaled_c2}

    scaled_fused = rrf_fuse(scaled_channels, k=60)

    assert [doc for doc, _ in base_fused] == [doc for doc, _ in scaled_fused]
    for (_, s_base), (_, s_scaled) in zip(base_fused, scaled_fused):
        assert abs(s_base - s_scaled) < 1e-9


def test_weighted_rrf():
    c1 = [("d1", 10.0), ("d2", 5.0)]
    c2 = [("d2", 1.0), ("d1", 0.5)]

    # Heavy weight on c1 favors d1
    fused_c1 = rrf_fuse({"c1": c1, "c2": c2}, k=60, weights={"c1": 5.0, "c2": 1.0})
    assert fused_c1[0][0] == "d1"

    # Heavy weight on c2 favors d2
    fused_c2 = rrf_fuse({"c1": c1, "c2": c2}, k=60, weights={"c1": 1.0, "c2": 5.0})
    assert fused_c2[0][0] == "d2"


def test_score_fusion_convex_combination():
    bm25 = [("d1", 10.0), ("d2", 0.0)]
    dense = [("d1", 0.0), ("d2", 10.0)]

    # Alpha 1.0: purely BM25 -> d1 first
    fused_1 = score_fuse(bm25, dense, alpha=1.0)
    assert fused_1[0][0] == "d1"
    assert abs(dict(fused_1)["d1"] - 1.0) < 1e-6
    assert abs(dict(fused_1)["d2"] - 0.0) < 1e-6

    # Alpha 0.0: purely Dense -> d2 first
    fused_0 = score_fuse(bm25, dense, alpha=0.0)
    assert fused_0[0][0] == "d2"
    assert abs(dict(fused_0)["d2"] - 1.0) < 1e-6
    assert abs(dict(fused_0)["d1"] - 0.0) < 1e-6

    # Alpha 0.5: equal -> tie
    fused_mid = score_fuse(bm25, dense, alpha=0.5)
    s_map = dict(fused_mid)
    assert abs(s_map["d1"] - 0.5) < 1e-6
    assert abs(s_map["d2"] - 0.5) < 1e-6
