import json

import numpy as np

from rag_bench.stats import compare_systems


def test_seed_determinism():
    """Verify that using the same fixed seed produces byte-identical results."""
    rng1 = np.random.default_rng(42)
    s1_a = rng1.normal(0.7, 0.1, 100).tolist()
    s1_b = rng1.normal(0.5, 0.1, 100).tolist()
    res1 = compare_systems(s1_a, s1_b, num_samples=1000, seed=42)
    json1 = json.dumps(res1, sort_keys=True)

    rng2 = np.random.default_rng(42)
    s2_a = rng2.normal(0.7, 0.1, 100).tolist()
    s2_b = rng2.normal(0.5, 0.1, 100).tolist()
    res2 = compare_systems(s2_a, s2_b, num_samples=1000, seed=42)
    json2 = json.dumps(res2, sort_keys=True)

    assert json1 == json2
    assert res1 == res2
