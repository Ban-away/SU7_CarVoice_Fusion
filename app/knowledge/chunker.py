"""Semantic text chunker for document ingestion.

Splits long documents into smaller, semantically coherent chunks while
preserving context overlap and metadata.
"""

import logging
import re
from typing import Optional

from app.knowledge.models import RetrievedDoc

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Split documents into overlapping semantic chunks.

    Parameters:
        chunk_size: Target number of characters per chunk (before overlap).
        chunk_overlap: Number of characters to overlap between adjacent chunks.
        separators: Regex patterns used to find natural split points,
            tried in order.  The first matching separator identifies
            the preferred split location near the chunk boundary.
        min_chunk_size: Chunks shorter than this are merged with the
            previous chunk (unless it is the only chunk).
    """

    _DEFAULT_SEPARATORS: list[str] = [
        r"\n\n+",  # paragraph breaks
        r"(?<=[。！？.!?])\s*",  # sentence boundaries (Chinese + English)
        r"[，,；;：:]\s*",  # clause boundaries
        r"\s{2,}",  # multiple spaces
        r"\s",  # single space (last resort)
    ]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
        min_chunk_size: int = 100,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or self._DEFAULT_SEPARATORS
        self._min_chunk_size = min_chunk_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_text(
        self,
        text: str,
        source: str = "unknown",
        start_page: int | None = None,
    ) -> list[RetrievedDoc]:
        """Split *text* into overlapping chunks, each wrapped as a RetrievedDoc.

        Args:
            text: The full document text.
            source: Source identifier for all produced chunks.
            start_page: Page number for the first chunk (incremented for
                subsequent pages when page breaks are detected).

        Returns:
            List of RetrievedDoc chunks.
        """
        if not text.strip():
            return []

        segments = self._split(text)
        docs: list[RetrievedDoc] = []
        current_page = start_page or 1

        for i, segment in enumerate(segments):
            # Try to detect page breaks (PDF page markers like \f or explicit markers)
            if "\f" in segment:
                sub_segments = segment.split("\f")
                for j, sub in enumerate(sub_segments):
                    if sub.strip():
                        docs.append(
                            RetrievedDoc(
                                content=sub.strip(),
                                source=source,
                                page=current_page,
                            )
                        )
                    if j < len(sub_segments) - 1:
                        current_page += 1  # form feed = page break
            else:
                docs.append(
                    RetrievedDoc(
                        content=segment.strip(),
                        source=source,
                        page=current_page,
                    )
                )

        # Merge undersized chunks
        docs = self._merge_small_chunks(docs)

        logger.debug(
            "Chunked text into %d segments (size=%d, overlap=%d)",
            len(docs),
            self._chunk_size,
            self._chunk_overlap,
        )
        return docs

    def chunk_documents(
        self,
        documents: list[RetrievedDoc],
    ) -> list[RetrievedDoc]:
        """Chunk a list of :class:`RetrievedDoc` instances by their content.

        Each input document's metadata (source, page) is preserved in
        the resulting chunks.

        Args:
            documents: Input documents to chunk.

        Returns:
            A new list of chunked RetrievedDoc instances.
        """
        all_chunks: list[RetrievedDoc] = []
        for doc in documents:
            chunks = self.chunk_text(
                text=doc.content,
                source=doc.source,
                start_page=doc.page,
            )
            all_chunks.extend(chunks)
        return all_chunks

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _split(self, text: str) -> list[str]:
        """Recursively split *text* into chunks at natural boundaries."""
        if len(text) <= self._chunk_size:
            return [text]

        # Find the best split point
        split_point = self._find_split_point(text)
        if split_point is None:
            # Force split at chunk boundary
            split_point = self._chunk_size

        # Make two segments with overlap, but ensure forward progress
        overlap = min(self._chunk_overlap // 2, split_point // 2)
        first = text[: split_point + overlap].strip()
        second = text[split_point - overlap :].strip()

        # Guard against infinite recursion: ensure each half is strictly shorter
        if len(first) >= len(text) or len(second) >= len(text):
            # Fallback: force split at exact chunk_size
            first = text[: self._chunk_size].strip()
            second = text[self._chunk_size :].strip()

        # Recurse if segment is still too long
        first_parts = self._split(first) if len(first) > self._chunk_size else [first]
        second_parts = (
            self._split(second) if len(second) > self._chunk_size else [second]
        )

        return first_parts + second_parts

    def _find_split_point(self, text: str) -> Optional[int]:
        """Find the best natural split point near ``chunk_size``."""
        window_start = max(0, self._chunk_size - self._chunk_overlap)
        window_end = min(len(text), self._chunk_size + self._chunk_overlap)
        search_text = text[window_start:window_end]

        for pattern in self._separators:
            matches = list(re.finditer(pattern, search_text))
            if not matches:
                continue
            # Pick the match closest to the target boundary
            target = self._chunk_size - window_start
            best_match = min(matches, key=lambda m: abs(m.end() - target))
            return window_start + best_match.end()

        return None

    @staticmethod
    def _merge_small_chunks(docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        """Merge chunks shorter than min_chunk_size with the previous chunk."""
        if not docs:
            return docs

        threshold = 100
        merged: list[RetrievedDoc] = []

        for doc in docs:
            if (
                merged
                and len(doc.content) < threshold
                and len(merged[-1].content) < 2000
            ):
                prev = merged[-1]
                merged[-1] = RetrievedDoc(
                    content=f"{prev.content} {doc.content}",
                    source=prev.source,
                    page=prev.page,
                )
            else:
                merged.append(doc)

        return merged
