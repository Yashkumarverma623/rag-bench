"""Statistical significance testing for IR system comparisons.

Implements paired bootstrap confidence intervals, paired permutation
randomization tests, and family-wise Holm-Bonferroni p-value corrections.
"""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def paired_bootstrap_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    num_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float, bool]:
    """Compute 95% percentile bootstrap confidence interval on mean paired difference (a - b).

    Returns:
        (mean_diff, ci_lower, ci_upper, excludes_zero)
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError(f"Lengths must match: {len(a)} vs {len(b)}")
    if len(a) == 0:
        return 0.0, 0.0, 0.0, False

    diffs = a - b
    mean_diff = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    n = len(diffs)
    # Generate bootstrap sample indices (num_resamples, n)
    indices = rng.integers(0, n, size=(num_resamples, n))
    bootstrap_means = np.mean(diffs[indices], axis=1)

    ci_lower = float(np.percentile(bootstrap_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_means, 100.0 * (1.0 - alpha / 2.0)))
    excludes_zero = bool((ci_lower > 0.0) or (ci_upper < 0.0))

    return mean_diff, ci_lower, ci_upper, excludes_zero


def paired_permutation_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    num_permutations: int = 10000,
    seed: int = 42,
) -> float:
    """Compute two-sided p-value using paired sign-flip permutation test."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError(f"Lengths must match: {len(a)} vs {len(b)}")
    if len(a) == 0:
        return 1.0

    diffs = a - b
    observed_stat = abs(float(np.mean(diffs)))
    if observed_stat == 0.0:
        return 1.0

    rng = np.random.default_rng(seed)
    n = len(diffs)
    # Signs: -1 or +1 with equal probability
    signs = rng.choice([-1.0, 1.0], size=(num_permutations, n))
    permuted_means = np.abs(np.mean(signs * diffs, axis=1))

    # Two-sided p-value with standard +1 correction
    extreme_count = int(np.sum(permuted_means >= observed_stat))
    p_val = float((1.0 + extreme_count) / (1.0 + num_permutations))
    return min(1.0, p_val)


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Apply Holm-Bonferroni step-down procedure to adjust p-values."""
    m = len(p_values)
    if m == 0:
        return []

    # Sort p-values with original indices
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted: List[Tuple[int, float]] = []

    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed_p, start=1):
        multiplier = m - rank + 1
        adj_p = min(1.0, float(p * multiplier))
        # Ensure monotonicity
        adj_p = max(adj_p, running_max)
        adj_p = min(1.0, adj_p)
        running_max = adj_p
        adjusted.append((orig_idx, adj_p))

    # Re-order back to original order
    adjusted.sort(key=lambda x: x[0])
    return [p for _, p in adjusted]


def compare_systems(
    scores_system: Sequence[float],
    scores_baseline: Sequence[float],
    num_samples: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Perform full pairwise statistical comparison between system and baseline."""
    mean_diff, ci_low, ci_high, excludes_zero = paired_bootstrap_ci(
        scores_system,
        scores_baseline,
        num_resamples=num_samples,
        seed=seed,
    )
    p_val = paired_permutation_test(
        scores_system,
        scores_baseline,
        num_permutations=num_samples,
        seed=seed,
    )
    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "ci_excludes_zero": excludes_zero,
        "raw_p_value": p_val,
    }


def adjust_comparisons(comparisons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Adjust p-values across a family of comparisons and determine significance."""
    raw_p_vals = [c["raw_p_value"] for c in comparisons]
    adj_p_vals = holm_bonferroni(raw_p_vals)

    results: List[Dict[str, Any]] = []
    for c, adj_p in zip(comparisons, adj_p_vals):
        res = dict(c)
        res["adjusted_p_value"] = adj_p
        # Mark significant only if CI excludes 0 and adjusted p < 0.05
        is_significant = res["ci_excludes_zero"] and (adj_p < 0.05)
        res["significant"] = is_significant
        res["status"] = "significant" if is_significant else "not significant"
        results.append(res)

    return results
