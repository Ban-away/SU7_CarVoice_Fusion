"""Abstract base class for all retrievers."""

from abc import ABC, abstractmethod

from app.knowledge.models import RetrievedDoc


class BaseRetriever(ABC):
    """Abstract retriever interface.

    All retriever implementations (BM25, FAISS, hybrid) must subclass
    this and implement :meth:`retrieve`.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievedDoc]:
        """Retrieve the top-k most relevant documents for *query*.

        Args:
            query: The search query string.
            top_k: Maximum number of documents to return.

        Returns:
            A list of RetrievedDoc objects sorted by descending relevance.
        """
        ...
