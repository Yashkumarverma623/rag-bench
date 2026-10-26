"""Dense exact retriever using sentence-transformers and brute-force cosine matmul."""

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_bench.retrievers.base import BaseRetriever, format_doc_text

BGE_SMALL_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_SMALL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


class DenseExactRetriever(BaseRetriever):
    """Dense retriever computing exact cosine similarity via brute-force NumPy matmul."""

    def __init__(
        self,
        model_name: str = BGE_SMALL_MODEL_NAME,
        revision: str = BGE_SMALL_REVISION,
        batch_size: int = 64,
        device: str = "cpu",
        embeddings: np.ndarray | None = None,
        doc_ids: List[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.device = device

        self._model: SentenceTransformer | None = None
        self.doc_ids: List[str] = doc_ids or []
        self.embeddings: np.ndarray | None = embeddings

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loader for SentenceTransformer model."""
        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name,
                revision=self.revision,
                device=self.device,
            )
        return self._model

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        """Encode query strings with normalization."""
        emb = self.model.encode(
            list(queries),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32)

    def index(self, corpus: Mapping[str, Mapping[str, str] | str]) -> None:
        """Encode all documents in corpus and store normalized embeddings."""
        self.doc_ids = list(corpus.keys())
        if not self.doc_ids:
            self.embeddings = None
            return

        texts = [format_doc_text(corpus[doc_id]) for doc_id in self.doc_ids]
        raw_emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self.embeddings = raw_emb.astype(np.float32)

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Retrieve top_k documents via exact cosine similarity."""
        if self.embeddings is None or not self.doc_ids:
            return []

        q_emb = self.encode_queries([query])[0]  # shape (dim,)
        scores = np.dot(self.embeddings, q_emb)  # shape (num_docs,)

        actual_k = min(top_k, len(self.doc_ids))
        if len(self.doc_ids) > actual_k:
            top_part = np.argpartition(-scores, actual_k)[:actual_k]
            top_part_scores = scores[top_part]
            sort_order = np.argsort(-top_part_scores)
            final_indices = top_part[sort_order]
        else:
            final_indices = np.argsort(-scores)[:actual_k]

        return [
            (self.doc_ids[idx], float(scores[idx]))
            for idx in final_indices
        ]

    def batch_search(
        self,
        queries: Mapping[str, str],
        top_k: int = 100,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Accelerate batch search via single Q @ D.T matrix multiplication."""
        if self.embeddings is None or not self.doc_ids or not queries:
            return {qid: [] for qid in queries}

        qids = list(queries.keys())
        q_texts = [queries[qid] for qid in qids]
        q_embs = self.encode_queries(q_texts)  # shape (num_queries, dim)

        # Batch matmul: (num_queries, num_docs)
        similarity_matrix = np.matmul(q_embs, self.embeddings.T)

        results: Dict[str, List[Tuple[str, float]]] = {}
        actual_k = min(top_k, len(self.doc_ids))

        for q_idx, qid in enumerate(qids):
            scores = similarity_matrix[q_idx]
            if len(self.doc_ids) > actual_k:
                top_part = np.argpartition(-scores, actual_k)[:actual_k]
                sort_order = np.argsort(-scores[top_part])
                final_indices = top_part[sort_order]
            else:
                final_indices = np.argsort(-scores)[:actual_k]

            results[qid] = [
                (self.doc_ids[idx], float(scores[idx]))
                for idx in final_indices
            ]

        return results
