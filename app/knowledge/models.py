"""Data models for the knowledge / RAG pipeline."""

from dataclasses import dataclass, field


@dataclass
class RetrievedDoc:
    """A single retrieved document chunk with metadata.

    Attributes:
        content: The text content of the document chunk.
        source: Document source identifier (filename, url, etc.).
        page: Optional page number from the source document.
        score: Relevance score (higher = more relevant).
    """

    content: str
    source: str
    page: int | None = None
    score: float = 0.0


@dataclass
class RetrieveResult:
    """Result of a retrieval operation.

    Attributes:
        docs: List of retrieved documents, sorted by relevance.
        latency_ms: Retrieval latency in milliseconds.
    """

    docs: list[RetrievedDoc] = field(default_factory=list)
    latency_ms: int = 0
