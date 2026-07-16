"""Knowledge service data models."""

from dataclasses import dataclass


@dataclass
class RetrievedDoc:
    """A single retrieved document chunk with metadata."""

    content: str
    source: str
    page: int | None = None
    score: float = 0.0
