import numpy as np
from rank_bm25 import BM25Okapi

from rag_bench.retrievers.base import format_doc_text
from rag_bench.retrievers.bm25 import BM25Retriever, tokenize


def test_bm25_vs_rank_bm25_rankings(tiny_corpus):
    """Unit-test that BM25 rankings match rank-bm25 on a corpus."""
    doc_ids = list(tiny_corpus.keys())
    tokenized_docs = [tokenize(format_doc_text(tiny_corpus[d])) for d in doc_ids]

    # Reference implementation
    ref_bm25 = BM25Okapi(tokenized_docs, k1=1.2, b=0.75, epsilon=0.25)

    # Our custom scipy sparse CSR retriever
    our_bm25 = BM25Retriever(k1=1.2, b=0.75, epsilon=0.25)
    our_bm25.index(tiny_corpus)

    test_queries = [
        "inverted indices search engines",
        "cross encoders relevance",
        "health cardiovascular disease",
        "approximate nearest neighbor",
    ]

    for q in test_queries:
        q_tokens = tokenize(q)
        ref_scores = ref_bm25.get_scores(q_tokens)
        ref_ranking = [
            doc_ids[idx] for idx in np.argsort(-ref_scores) if ref_scores[idx] > 0
        ]

        our_results = our_bm25.search(q, top_k=len(doc_ids))
        our_ranking = [doc_id for doc_id, s in our_results if s > 0]

        # Top document must match identically
        assert our_ranking[0] == ref_ranking[0], f"Top doc mismatch for query '{q}'"

        # Scores for all documents must match ref scores within float tolerance
        our_score_map = {doc_id: s for doc_id, s in our_results}
        for idx, doc_id in enumerate(doc_ids):
            ref_s = ref_scores[idx]
            our_s = our_score_map.get(doc_id, 0.0)
            assert abs(ref_s - our_s) < 1e-5, f"Score mismatch on {doc_id}: {ref_s} vs {our_s}"


def test_bm25_empty_query_and_corpus():
    retriever = BM25Retriever()
    assert retriever.search("test") == []

    retriever.index({"d1": "hello world"})
    assert retriever.search("") == []
    res = retriever.search("hello")
    assert len(res) == 1
    assert res[0][0] == "d1"
