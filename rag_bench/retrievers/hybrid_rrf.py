"""Hybrid Reciprocal Rank Fusion (RRF) retriever."""

import collections
from typing import Dict, List, Mapping, Sequence, Tuple

from rag_bench.retrievers.base import BaseRetriever


def rrf_fuse(
    channel_rankings: Mapping[str, Sequence[Tuple[str, float]]],
    k: int = 60,
    weights: Mapping[str, float] | None = None,
    top_k: int = 100,
) -> List[Tuple[str, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Score(d) = sum_{c} w_c / (k + rank_c(d)) where rank is 1-indexed.
    """
    scores: Dict[str, float] = collections.defaultdict(float)
    active_weights = weights or {}

    for channel, ranked_list in channel_rankings.items():
        w = float(active_weights.get(channel, 1.0))
        for rank_1_based, (doc_id, _) in enumerate(ranked_list, start=1):
            scores[doc_id] += w / (k + rank_1_based)

    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_docs[:top_k]


class HybridRRFRetriever(BaseRetriever):
    """Hybrid retriever combining multiple first-stage retrievers via RRF."""

    def __init__(
        self,
        retrievers: Mapping[str, BaseRetriever],
        k: int = 60,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self.retrievers = dict(retrievers)
        self.k = k
        self.weights = weights or {ch: 1.0 for ch in self.retrievers}

    def index(self, corpus: Mapping[str, Mapping[str, str] | str]) -> None:
        """Index corpus into each underlying retriever."""
        for retriever in self.retrievers.values():
            retriever.index(corpus)

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Retrieve candidates from each channel and fuse using RRF."""
        # Retrieve candidates (fetch up to top_k * 2 or max 200 from each channel)
        fetch_k = max(top_k * 2, 200)
        channel_results: Dict[str, List[Tuple[str, float]]] = {}
        for channel, retriever in self.retrievers.items():
            channel_results[channel] = retriever.search(query, top_k=fetch_k)

        return rrf_fuse(channel_results, k=self.k, weights=self.weights, top_k=top_k)

    def batch_search(
        self,
        queries: Mapping[str, str],
        top_k: int = 100,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Batch retrieve from each channel and fuse."""
        fetch_k = max(top_k * 2, 200)
        channel_batch_results: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
        for channel, retriever in self.retrievers.items():
            channel_batch_results[channel] = retriever.batch_search(queries, top_k=fetch_k)

        final_results: Dict[str, List[Tuple[str, float]]] = {}
        for qid in queries:
            q_channel_rankings = {
                channel: channel_batch_results[channel].get(qid, [])
                for channel in self.retrievers
            }
            final_results[qid] = rrf_fuse(
                q_channel_rankings,
                k=self.k,
                weights=self.weights,
                top_k=top_k,
            )

        return final_results
