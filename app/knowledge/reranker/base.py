"""Abstract base class for rerankers."""

from abc import ABC, abstractmethod

from app.knowledge.models import RetrievedDoc


class BaseReranker(ABC):
    """Abstract reranker interface.

    Rerankers take a query and a list of candidate documents and return
    a reordered list, typically improving precision of the top-k results.
    """

    @abstractmethod
    def rerank(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        """Rerank *docs* by relevance to *query*.

        Args:
            query: The search query string.
            docs: Candidates to rerank (may already be scored).

        Returns:
            Documents reordered by descending relevance.
        """
        ...
