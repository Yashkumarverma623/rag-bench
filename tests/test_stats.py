import numpy as np

from rag_bench.stats import (
    adjust_comparisons,
    compare_systems,
    holm_bonferroni,
    paired_bootstrap_ci,
    paired_permutation_test,
)


def test_bootstrap_synthetic_known_difference():
    """CI excludes 0 for a synthetic known-difference set."""
    rng = np.random.default_rng(42)
    # System A clearly outperforms System B by ~0.3
    sys_a = rng.normal(loc=0.75, scale=0.08, size=100)
    sys_b = rng.normal(loc=0.45, scale=0.08, size=100)

    mean_diff, ci_low, ci_high, excludes_zero = paired_bootstrap_ci(
        sys_a, sys_b, num_resamples=10000, seed=42
    )

    assert excludes_zero is True
    assert ci_low > 0.0
    assert ci_high > ci_low
    assert abs(mean_diff - 0.3) < 0.05


def test_bootstrap_synthetic_null_difference():
    """CI includes 0 for a no-difference set."""
    rng = np.random.default_rng(42)
    # System A and System B drawn from same distribution
    base = rng.normal(loc=0.5, scale=0.1, size=150)
    sys_a = base + rng.normal(loc=0.0, scale=0.02, size=150)
    sys_b = base + rng.normal(loc=0.0, scale=0.02, size=150)

    mean_diff, ci_low, ci_high, excludes_zero = paired_bootstrap_ci(
        sys_a, sys_b, num_resamples=10000, seed=42
    )

    assert excludes_zero is False
    assert ci_low < 0.0 < ci_high


def test_permutation_test_known_difference():
    rng = np.random.default_rng(42)
    sys_a = rng.normal(loc=0.7, scale=0.05, size=80)
    sys_b = rng.normal(loc=0.4, scale=0.05, size=80)

    p_val = paired_permutation_test(sys_a, sys_b, num_permutations=5000, seed=42)
    assert p_val < 0.001


def test_permutation_test_null():
    rng = np.random.default_rng(42)
    sys_a = rng.normal(loc=0.5, scale=0.1, size=100)
    sys_b = rng.normal(loc=0.5, scale=0.1, size=100)

    p_val = paired_permutation_test(sys_a, sys_b, num_permutations=5000, seed=42)
    assert p_val > 0.05


def test_holm_bonferroni_correction():
    # Example raw p-values
    p_vals = [0.001, 0.01, 0.03, 0.04, 0.5]
    # m = 5
    # rank 1: 0.001 * 5 = 0.005
    # rank 2: 0.01 * 4 = 0.04
    # rank 3: 0.03 * 3 = 0.09
    # rank 4: 0.04 * 2 = 0.08 -> running max ensures it becomes >= 0.09 -> 0.09
    # rank 5: 0.5 * 1 = 0.5 -> 0.5
    adj = holm_bonferroni(p_vals)
    assert len(adj) == 5
    assert abs(adj[0] - 0.005) < 1e-6
    assert abs(adj[1] - 0.04) < 1e-6
    assert abs(adj[2] - 0.09) < 1e-6
    assert abs(adj[3] - 0.09) < 1e-6
    assert abs(adj[4] - 0.5) < 1e-6

    # Test reordering preserves original indices
    unsorted_p = [0.04, 0.001, 0.5, 0.01, 0.03]
    unsorted_adj = holm_bonferroni(unsorted_p)
    assert abs(unsorted_adj[1] - 0.005) < 1e-6
    assert abs(unsorted_adj[3] - 0.04) < 1e-6


def test_compare_systems_and_adjust():
    rng = np.random.default_rng(42)
    s1 = rng.normal(0.8, 0.05, 50)
    s2 = rng.normal(0.5, 0.05, 50)
    s3 = rng.normal(0.51, 0.05, 50)
    base = rng.normal(0.5, 0.05, 50)

    c1 = compare_systems(s1, base, num_samples=2000, seed=42)
    c2 = compare_systems(s2, base, num_samples=2000, seed=42)
    c3 = compare_systems(s3, base, num_samples=2000, seed=42)

    adj = adjust_comparisons([c1, c2, c3])
    assert adj[0]["significant"] is True
    assert adj[0]["status"] == "significant"
    assert adj[1]["significant"] is False
    assert adj[1]["status"] == "not significant"
