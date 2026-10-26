from unittest.mock import MagicMock

from rag_bench.retrievers.reranker import CrossEncoderReranker


def test_reranker_top_n_candidate_selection(tiny_corpus):
    reranker = CrossEncoderReranker()
    # Mock model predict: return inverted scores
    mock_model = MagicMock()
    # predict returns higher score for whatever pairs were passed
    mock_model.predict.side_effect = lambda pairs, **kwargs: [float(len(p[1])) for p in pairs]
    reranker._model = mock_model

    candidates = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7), ("doc4", 0.6)]

    # Rerank with top_n=2
    reranked = reranker.rerank(
        query="test query",
        candidates=candidates,
        corpus=tiny_corpus,
        top_n=2,
        top_k=2,
    )

    assert len(reranked) == 2
    # Verify only top 2 candidates were sent to model
    assert mock_model.predict.call_count == 1
    call_pairs = mock_model.predict.call_args[0][0]
    assert len(call_pairs) == 2
