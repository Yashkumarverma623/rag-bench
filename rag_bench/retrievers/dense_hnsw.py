"""Dense HNSW retriever using Faiss IndexHNSWFlat with normalized cosine embeddings."""

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import faiss
import numpy as np

from rag_bench.retrievers.base import BaseRetriever
from rag_bench.retrievers.dense_exact import DenseExactRetriever


class DenseHNSWRetriever(BaseRetriever):
    """Dense retriever using FAISS IndexHNSWFlat for Approximate Nearest Neighbor search."""

    def __init__(
        self,
        exact_retriever: DenseExactRetriever | None = None,
        m: int = 32,
        ef_construction: int = 64,
        ef_search: int = 64,
    ) -> None:
        self.exact_retriever = exact_retriever or DenseExactRetriever()
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self.index_hnsw: faiss.IndexHNSWFlat | None = None
        self.doc_ids: List[str] = []

    def set_ef_search(self, ef_search: int) -> None:
        """Dynamically set efSearch parameter for search queries."""
        self.ef_search = ef_search
        if self.index_hnsw is not None:
            self.index_hnsw.hnsw.efSearch = ef_search

    def index(self, corpus: Mapping[str, Mapping[str, str] | str]) -> None:
        """Encode corpus via exact retriever and populate Faiss HNSW index."""
        if self.exact_retriever.embeddings is None:
            self.exact_retriever.index(corpus)

        self.doc_ids = list(self.exact_retriever.doc_ids)
        if not self.doc_ids or self.exact_retriever.embeddings is None:
            self.index_hnsw = None
            return

        dim = self.exact_retriever.embeddings.shape[1]
        # FAISS IndexHNSWFlat with METRIC_INNER_PRODUCT (exact cosine on normalized vectors)
        index = faiss.IndexHNSWFlat(dim, self.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search

        # Add normalized vectors
        index.add(self.exact_retriever.embeddings)
        self.index_hnsw = index

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Retrieve top_k documents using HNSW approximate search."""
        if self.index_hnsw is None or not self.doc_ids:
            return []

        q_emb = self.exact_retriever.encode_queries([query])  # shape (1, dim)
        self.index_hnsw.hnsw.efSearch = self.ef_search

        actual_k = min(top_k, len(self.doc_ids))
        distances, indices = self.index_hnsw.search(q_emb, actual_k)

        res: List[Tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                res.append((self.doc_ids[idx], float(dist)))
        return res

    def batch_search(
        self,
        queries: Mapping[str, str],
        top_k: int = 100,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Accelerated batch ANN search using FAISS multi-query search."""
        if self.index_hnsw is None or not self.doc_ids or not queries:
            return {qid: [] for qid in queries}

        qids = list(queries.keys())
        q_texts = [queries[qid] for qid in qids]
        q_embs = self.exact_retriever.encode_queries(q_texts)

        self.index_hnsw.hnsw.efSearch = self.ef_search
        actual_k = min(top_k, len(self.doc_ids))
        distances, indices = self.index_hnsw.search(q_embs, actual_k)

        results: Dict[str, List[Tuple[str, float]]] = {}
        for q_idx, qid in enumerate(qids):
            q_res: List[Tuple[str, float]] = []
            for dist, idx in zip(distances[q_idx], indices[q_idx]):
                if idx >= 0:
                    q_res.append((self.doc_ids[idx], float(dist)))
            results[qid] = q_res

        return results

    def sweep_ef_search(
        self,
        queries: Mapping[str, str],
        exact_results: Mapping[str, Sequence[Tuple[str, float]]],
        qrels: Mapping[str, Mapping[str, float]],
        ef_search_values: Sequence[int] = (10, 20, 40, 64, 128, 256),
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Sweep efSearch and compute recall@top_k vs exact search, and nDCG@10 loss."""
        from rag_bench.metrics import evaluate_run

        # Compute exact nDCG@10 baseline
        exact_mean, _ = evaluate_run(qrels, exact_results)
        exact_ndcg = exact_mean.get("ndcg@10", 0.0)

        sweep_report: List[Dict[str, Any]] = []

        for ef in ef_search_values:
            self.set_ef_search(ef)
            hnsw_results = self.batch_search(queries, top_k=top_k)

            # Compute recall@10 vs exact search:
            # For each query, what fraction of exact top_k docs are found in HNSW top_k?
            recalls: List[float] = []
            for qid in queries:
                exact_top_ids = set([doc_id for doc_id, _ in exact_results.get(qid, [])[:top_k]])
                hnsw_top_ids = set([doc_id for doc_id, _ in hnsw_results.get(qid, [])[:top_k]])
                if exact_top_ids:
                    recalls.append(len(exact_top_ids & hnsw_top_ids) / len(exact_top_ids))
                else:
                    recalls.append(1.0)

            mean_recall_vs_exact = float(np.mean(recalls)) if recalls else 0.0

            # Compute nDCG@10 loss vs exact
            hnsw_mean, _ = evaluate_run(qrels, hnsw_results)
            hnsw_ndcg = hnsw_mean.get("ndcg@10", 0.0)
            ndcg_loss = exact_ndcg - hnsw_ndcg

            sweep_report.append({
                "ef_search": ef,
                "ann_recall_at_10_vs_exact": mean_recall_vs_exact,
                "ndcg_at_10": hnsw_ndcg,
                "ndcg_loss_vs_exact": ndcg_loss,
            })

        return sweep_report
