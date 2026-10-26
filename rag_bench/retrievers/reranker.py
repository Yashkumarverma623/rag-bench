"""Cross-encoder reranker using sentence-transformers CrossEncoder."""

from typing import Dict, List, Mapping, Sequence, Tuple

from sentence_transformers import CrossEncoder

from rag_bench.retrievers.base import BaseRetriever, format_doc_text

CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


class CrossEncoderReranker:
    """Reranks top-N candidates using a CrossEncoder model."""

    def __init__(
        self,
        model_name: str = CROSS_ENCODER_MODEL_NAME,
        revision: str = CROSS_ENCODER_REVISION,
        max_length: int = 512,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy load CrossEncoder model."""
        if self._model is None:
            self._model = CrossEncoder(
                self.model_name,
                revision=self.revision,
                max_length=self.max_length,
                device=self.device,
            )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[Tuple[str, float]],
        corpus: Mapping[str, Mapping[str, str] | str],
        top_n: int = 100,
        top_k: int = 100,
    ) -> List[Tuple[str, float]]:
        """Rerank top_n candidates from first stage and return up to top_k results."""
        if not candidates:
            return []

        # Slice to top_n candidates
        rerank_pool = list(candidates[:top_n])
        pairs: List[Tuple[str, str]] = []
        valid_doc_ids: List[str] = []

        for doc_id, _ in rerank_pool:
            if doc_id in corpus:
                doc_text = format_doc_text(corpus[doc_id])
                pairs.append((query, doc_text))
                valid_doc_ids.append(doc_id)

        if not pairs:
            return list(candidates[:top_k])

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        reranked = [
            (doc_id, float(s))
            for doc_id, s in zip(valid_doc_ids, scores)
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)

        # If candidates beyond top_n exist and top_k > top_n, append them after min score
        if len(candidates) > top_n and top_k > top_n:
            min_rerank_score = min(s for _, s in reranked) - 1.0
            remainder = [
                (doc_id, min_rerank_score - idx)
                for idx, (doc_id, _) in enumerate(candidates[top_n:top_k], start=1)
            ]
            return (reranked + remainder)[:top_k]

        return reranked[:top_k]


class RerankedRetriever(BaseRetriever):
    """Wrapper that composes a first-stage retriever with a cross-encoder reranker."""

    def __init__(
        self,
        first_stage: BaseRetriever,
        reranker: CrossEncoderReranker,
        corpus: Mapping[str, Mapping[str, str] | str],
        top_n: int = 100,
    ) -> None:
        self.first_stage = first_stage
        self.reranker = reranker
        self.corpus = corpus
        self.top_n = top_n

    def index(self, corpus: Mapping[str, Mapping[str, str] | str]) -> None:
        self.corpus = corpus
        self.first_stage.index(corpus)

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        first_stage_results = self.first_stage.search(query, top_k=max(self.top_n, top_k))
        return self.reranker.rerank(
            query=query,
            candidates=first_stage_results,
            corpus=self.corpus,
            top_n=self.top_n,
            top_k=top_k,
        )

    def batch_search(
        self,
        queries: Mapping[str, str],
        top_k: int = 100,
    ) -> Dict[str, List[Tuple[str, float]]]:
        first_stage_batch = self.first_stage.batch_search(
            queries, top_k=max(self.top_n, top_k)
        )
        results: Dict[str, List[Tuple[str, float]]] = {}
        for qid, query_text in queries.items():
            results[qid] = self.reranker.rerank(
                query=query_text,
                candidates=first_stage_batch.get(qid, []),
                corpus=self.corpus,
                top_n=self.top_n,
                top_k=top_k,
            )
        return results
