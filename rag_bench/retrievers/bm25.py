"""Okapi BM25 retriever implemented with a SciPy sparse CSR inverted index."""

import collections
import math
import re
from typing import Dict, List, Mapping, Tuple

import numpy as np
import scipy.sparse as sp

from rag_bench.retrievers.base import BaseRetriever, format_doc_text


def tokenize(text: str) -> List[str]:
    """Lowercase + regex word split tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever(BaseRetriever):
    """Okapi BM25 retriever using a SciPy CSR sparse matrix inverted index."""

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.doc_ids: List[str] = []
        self.doc_id_to_idx: Dict[str, int] = {}
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.matrix: sp.csr_matrix | None = None
        self.avgdl: float = 0.0
        self.doc_lens: np.ndarray | None = None

    def index(self, corpus: Mapping[str, Mapping[str, str] | str]) -> None:
        """Build the sparse inverted index for the given corpus."""
        self.doc_ids = list(corpus.keys())
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        n_docs = len(self.doc_ids)
        if n_docs == 0:
            return

        tokenized_docs: List[List[str]] = []
        doc_lens_list: List[int] = []
        df: collections.Counter[str] = collections.Counter()

        self.vocab = {}
        for doc_id in self.doc_ids:
            raw_text = format_doc_text(corpus[doc_id])
            tokens = tokenize(raw_text)
            tokenized_docs.append(tokens)
            doc_lens_list.append(len(tokens))
            unique_tokens = set(tokens)
            df.update(unique_tokens)
            for t in unique_tokens:
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab)

        self.doc_lens = np.array(doc_lens_list, dtype=np.float64)
        total_tokens = float(np.sum(self.doc_lens))
        self.avgdl = total_tokens / n_docs if n_docs > 0 else 0.0

        # Compute IDF with negative floor matching rank-bm25
        idf_sum = 0.0
        negative_idfs: List[str] = []
        self.idf = {}
        for word, freq in df.items():
            val = math.log(n_docs - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = val
            idf_sum += val
            if val < 0.0:
                negative_idfs.append(word)

        if self.idf:
            average_idf = idf_sum / len(self.idf)
            floor_val = max(1e-6, self.epsilon * average_idf)
            for word in negative_idfs:
                self.idf[word] = floor_val

        # Precompute length normalization term: k1 * (1 - b + b * (dl / avgdl))
        if self.avgdl > 0.0:
            len_norm = self.k1 * (1.0 - self.b + self.b * (self.doc_lens / self.avgdl))
        else:
            len_norm = np.full(n_docs, self.k1)

        # Build sparse CSR matrix entries: row=doc_idx, col=word_idx, val=bm25_weight
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        k1_plus_one = self.k1 + 1.0
        for d_idx, tokens in enumerate(tokenized_docs):
            term_counts = collections.Counter(tokens)
            d_norm = len_norm[d_idx]
            for term, count in term_counts.items():
                col_idx = self.vocab[term]
                w_idf = self.idf[term]
                weight = w_idf * (count * k1_plus_one) / (count + d_norm)
                rows.append(d_idx)
                cols.append(col_idx)
                data.append(weight)

        self.matrix = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(n_docs, len(self.vocab)),
            dtype=np.float32,
        )

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Retrieve top_k documents for a single query using sparse dot product."""
        if self.matrix is None or not self.doc_ids:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        q_counts = collections.Counter(tokens)
        query_cols: List[int] = []
        query_weights: List[float] = []

        for t, count in q_counts.items():
            if t in self.vocab:
                query_cols.append(self.vocab[t])
                query_weights.append(float(count))

        if not query_cols:
            return []

        # Construct sparse column vector for query
        q_vec = sp.csr_matrix(
            (query_weights, (query_cols, [0] * len(query_cols))),
            shape=(len(self.vocab), 1),
            dtype=np.float32,
        )

        # Compute all document scores via sparse matrix-vector dot product
        scores: np.ndarray = self.matrix.dot(q_vec).toarray().flatten()

        # Find non-zero scoring documents
        non_zero_indices = np.flatnonzero(scores > 0.0)
        if non_zero_indices.size == 0:
            return []

        actual_k = min(top_k, non_zero_indices.size)
        candidate_scores = scores[non_zero_indices]

        # Use argpartition for fast top_k selection
        if candidate_scores.size > actual_k:
            part_idx = np.argpartition(-candidate_scores, actual_k)[:actual_k]
            top_indices = non_zero_indices[part_idx]
            top_scores = scores[top_indices]
            sort_order = np.argsort(-top_scores)
            final_indices = top_indices[sort_order]
            final_scores = top_scores[sort_order]
        else:
            sort_order = np.argsort(-candidate_scores)
            final_indices = non_zero_indices[sort_order]
            final_scores = candidate_scores[sort_order]

        return [
            (self.doc_ids[idx], float(score))
            for idx, score in zip(final_indices, final_scores)
        ]
