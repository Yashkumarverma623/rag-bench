"""Hybrid Score Fusion retriever using min-max normalization and convex combination."""

from typing import Dict, List, Mapping, Sequence, Tuple

from rag_bench.retrievers.base import BaseRetriever


def score_fuse(
    bm25_ranked: Sequence[Tuple[str, float]],
    dense_ranked: Sequence[Tuple[str, float]],
    alpha: float = 0.5,
    top_k: int = 100,
) -> List[Tuple[str, float]]:
    """Min-max normalize per query channel scores and compute convex combination.

    S(d) = alpha * S'_bm25(d) + (1 - alpha) * S'_dense(d)
    """
    # 1. Normalize BM25 scores
    norm_bm25: Dict[str, float] = {}
    if bm25_ranked:
        bm25_scores = [s for _, s in bm25_ranked]
        min_b = min(bm25_scores)
        max_b = max(bm25_scores)
        range_b = max_b - min_b
        for doc_id, s in bm25_ranked:
            norm_bm25[doc_id] = (s - min_b) / range_b if range_b > 1e-9 else 1.0

    # 2. Normalize Dense scores
    norm_dense: Dict[str, float] = {}
    if dense_ranked:
        dense_scores = [s for _, s in dense_ranked]
        min_d = min(dense_scores)
        max_d = max(dense_scores)
        range_d = max_d - min_d
        for doc_id, s in dense_ranked:
            norm_dense[doc_id] = (s - min_d) / range_d if range_d > 1e-9 else 1.0

    # 3. Combine scores across all unique documents
    all_doc_ids = set(norm_bm25.keys()) | set(norm_dense.keys())
    combined_scores: Dict[str, float] = {}

    for doc_id in all_doc_ids:
        s_bm25 = norm_bm25.get(doc_id, 0.0)
        s_dense = norm_dense.get(doc_id, 0.0)
        combined_scores[doc_id] = alpha * s_bm25 + (1.0 - alpha) * s_dense

    sorted_docs = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_docs[:top_k]


class HybridScoreFusionRetriever(BaseRetriever):
    """Retriever combining BM25 and Dense channels via min-max score fusion."""

    def __init__(
        self,
        bm25_retriever: BaseRetriever,
        dense_retriever: BaseRetriever,
        alpha: float = 0.5,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.alpha = alpha

    def index(self, corpus: Mapping[str, Mapping[str, str] | str]) -> None:
        """Index corpus in both retrievers."""
        self.bm25_retriever.index(corpus)
        self.dense_retriever.index(corpus)

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Retrieve from both channels and fuse with min-max convex combination."""
        fetch_k = max(top_k * 2, 200)
        bm25_res = self.bm25_retriever.search(query, top_k=fetch_k)
        dense_res = self.dense_retriever.search(query, top_k=fetch_k)
        return score_fuse(bm25_res, dense_res, alpha=self.alpha, top_k=top_k)

    def batch_search(
        self,
        queries: Mapping[str, str],
        top_k: int = 100,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Batch retrieve from both channels and fuse."""
        fetch_k = max(top_k * 2, 200)
        bm25_batch = self.bm25_retriever.batch_search(queries, top_k=fetch_k)
        dense_batch = self.dense_retriever.batch_search(queries, top_k=fetch_k)

        final_results: Dict[str, List[Tuple[str, float]]] = {}
        for qid in queries:
            final_results[qid] = score_fuse(
                bm25_batch.get(qid, []),
                dense_batch.get(qid, []),
                alpha=self.alpha,
                top_k=top_k,
            )
        return final_results
