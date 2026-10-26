"""Base Retriever interface for rag-bench."""

from abc import ABC, abstractmethod
from typing import Dict, List, Mapping, Tuple


def format_doc_text(doc: Mapping[str, str] | str) -> str:
    """Format BEIR document as 'title + ' ' + text' without chunking."""
    if isinstance(doc, str):
        return doc
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    if title and text:
        return f"{title} {text}"
    return title or text


class BaseRetriever(ABC):
    """Abstract base class for all retrieval pipelines."""

    @abstractmethod
    def index(self, corpus: Mapping[str, Mapping[str, str] | str]) -> None:
        """Build or prepare index for the given corpus.

        Corpus is a mapping from doc_id to either document dictionary
        ({'title': ..., 'text': ...}) or preformatted string.
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Retrieve top_k documents for a single query.

        Returns a list of (doc_id, score) sorted descending by score.
        """
        pass

    def batch_search(
        self,
        queries: Mapping[str, str],
        top_k: int = 100,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Retrieve top_k documents for multiple queries.

        Default implementation iterates over queries. Subclasses can override
        for batch acceleration.
        """
        results: Dict[str, List[Tuple[str, float]]] = {}
        for qid, query_text in queries.items():
            results[qid] = self.search(query_text, top_k=top_k)
        return results
