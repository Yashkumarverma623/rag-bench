import faiss
import numpy as np

from rag_bench.retrievers.dense_exact import DenseExactRetriever


def test_hnsw_recall_vs_exact_synthetic():
    """Verify HNSW recall vs exact search on a synthetic normalized vector set."""
    rng = np.random.default_rng(42)
    num_docs = 1000
    dim = 64
    num_queries = 25
    top_k = 10

    # Generate synthetic random unit vectors
    raw_docs = rng.normal(size=(num_docs, dim)).astype(np.float32)
    doc_embs = raw_docs / np.linalg.norm(raw_docs, axis=1, keepdims=True)

    raw_queries = rng.normal(size=(num_queries, dim)).astype(np.float32)
    q_embs = raw_queries / np.linalg.norm(raw_queries, axis=1, keepdims=True)

    # 1. Exact ground truth via inner product matmul
    exact_sim = np.matmul(q_embs, doc_embs.T)
    exact_top_k = [
        set(np.argsort(-exact_sim[i])[:top_k])
        for i in range(num_queries)
    ]

    # 2. HNSW index via Faiss IndexHNSWFlat
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 64
    index.add(doc_embs)

    # Test across efSearch sweep
    for ef in [10, 40, 64, 128]:
        index.hnsw.efSearch = ef
        _, indices = index.search(q_embs, top_k)

        recalls = []
        for i in range(num_queries):
            hnsw_set = set(indices[i])
            recalls.append(len(exact_top_k[i] & hnsw_set) / top_k)

        mean_recall = np.mean(recalls)
        # HNSW recall should improve or stay high
        if ef >= 64:
            assert mean_recall >= 0.95, f"HNSW recall {mean_recall} too low at efSearch={ef}"


def test_dense_exact_ranking_order():
    """Verify exact cosine matmul correctly ranks orthogonal and identical vectors."""
    # d1 identical to q, d2 orthogonal, d3 opposite
    q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    docs = np.array([
        [1.0, 0.0, 0.0, 0.0],   # doc1: cosine = 1.0
        [0.0, 1.0, 0.0, 0.0],   # doc2: cosine = 0.0
        [-1.0, 0.0, 0.0, 0.0],  # doc3: cosine = -1.0
    ], dtype=np.float32)

    retriever = DenseExactRetriever(
        embeddings=docs,
        doc_ids=["doc1", "doc2", "doc3"],
    )
    # Mock encode_queries
    retriever.encode_queries = lambda q_list: q  # type: ignore[assignment]

    results = retriever.search("dummy", top_k=3)
    assert [doc_id for doc_id, _ in results] == ["doc1", "doc2", "doc3"]
    assert abs(results[0][1] - 1.0) < 1e-6
    assert abs(results[1][1] - 0.0) < 1e-6
    assert abs(results[2][1] - (-1.0)) < 1e-6
