"""Online latency profiling and batch throughput measurement per pipeline stage."""

import time
from typing import Any, Callable, Dict, List

import numpy as np


class LatencyTracker:
    """Tracks latency measurements across pipeline stages in milliseconds."""

    def __init__(self) -> None:
        self.stage_timings: Dict[str, List[float]] = {}

    def record(self, stage: str, duration_ms: float) -> None:
        if stage not in self.stage_timings:
            self.stage_timings[stage] = []
        self.stage_timings[stage].append(duration_ms)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Calculate p50, p95, p99, mean, min, max per stage."""
        stats: Dict[str, Dict[str, float]] = {}
        for stage, durations in self.stage_timings.items():
            if not durations:
                continue
            arr = np.array(durations, dtype=np.float64)
            stats[stage] = {
                "p50_ms": float(np.percentile(arr, 50.0)),
                "p95_ms": float(np.percentile(arr, 95.0)),
                "p99_ms": float(np.percentile(arr, 99.0)),
                "mean_ms": float(np.mean(arr)),
                "min_ms": float(np.min(arr)),
                "max_ms": float(np.max(arr)),
                "sample_count": len(arr),
            }
        return stats


def measure_batch_throughput(
    search_fn: Callable[[], Any],
    num_queries: int,
) -> Dict[str, float]:
    """Measure batch throughput (Queries Per Second, QPS)."""
    start_time = time.perf_counter()
    search_fn()
    elapsed_sec = time.perf_counter() - start_time
    qps = num_queries / elapsed_sec if elapsed_sec > 0 else 0.0
    return {
        "elapsed_sec": float(elapsed_sec),
        "total_queries": num_queries,
        "queries_per_sec": float(qps),
    }
