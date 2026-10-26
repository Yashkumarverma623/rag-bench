"""Retriever implementations for rag-bench."""

from rag_bench.retrievers.base import BaseRetriever, format_doc_text
from rag_bench.retrievers.bm25 import BM25Retriever
from rag_bench.retrievers.dense_exact import DenseExactRetriever
from rag_bench.retrievers.dense_hnsw import DenseHNSWRetriever
from rag_bench.retrievers.hybrid_rrf import HybridRRFRetriever, rrf_fuse
from rag_bench.retrievers.hybrid_score_fusion import HybridScoreFusionRetriever, score_fuse
from rag_bench.retrievers.reranker import CrossEncoderReranker, RerankedRetriever

__all__ = [
    "BaseRetriever",
    "format_doc_text",
    "BM25Retriever",
    "DenseExactRetriever",
    "DenseHNSWRetriever",
    "HybridRRFRetriever",
    "rrf_fuse",
    "HybridScoreFusionRetriever",
    "score_fuse",
    "CrossEncoderReranker",
    "RerankedRetriever",
]
