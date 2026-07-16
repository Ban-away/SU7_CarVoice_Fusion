"""BM25 retriever with Chinese tokenization support.

Uses the ``rank_bm25`` library for scoring, with a fallback to simple
term-frequency scoring when the library is not installed.  Chinese text
is tokenized via jieba (if available) or character-level splitting.
"""

import logging
import re
from typing import Optional

from app.knowledge.models import RetrievedDoc
from app.knowledge.retriever.base import BaseRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

try:
    import jieba

    _HAS_JIEBA = True
except ImportError:  # pragma: no cover
    _HAS_JIEBA = False

try:
    from rank_bm25 import BM25Okapi

    _HAS_RANK_BM25 = True
except ImportError:  # pragma: no cover
    _HAS_RANK_BM25 = False


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _is_chinese(char: str) -> bool:
    """Return True if *char* is a CJK Unified Ideograph."""
    return "一" <= char <= "鿿"


def _tokenize(text: str) -> list[str]:
    """Tokenize *text* into a list of terms.

    Uses jieba for Chinese tokenization if available; otherwise falls
    back to character-level splitting for Chinese characters and
    whitespace splitting for non-Chinese tokens.
    """
    if _HAS_JIEBA:
        tokens = jieba.lcut(text)
        return [t.strip() for t in tokens if t.strip()]

    # Character-level fallback for Chinese
    tokens: list[str] = []
    buf = ""
    for ch in text:
        if _is_chinese(ch):
            if buf.strip():
                tokens.append(buf.strip())
                buf = ""
            tokens.append(ch)
        elif ch.isalnum():
            buf += ch
        else:
            if buf.strip():
                tokens.append(buf.strip())
                buf = ""
    if buf.strip():
        tokens.append(buf.strip())
    return tokens


# ---------------------------------------------------------------------------
# Simple TF scorer (fallback when rank_bm25 is unavailable)
# ---------------------------------------------------------------------------

class _SimpleTFScorer:
    """Minimal TF-based scorer for use when rank_bm25 is not installed."""

    def __init__(self, corpus_tokens: list[list[str]]) -> None:
        self._corpus = corpus_tokens
        self._doc_count = len(corpus_tokens)
        # Build term -> number-of-docs-containing-term
        self._idf: dict[str, float] = {}
        for doc_tokens in corpus_tokens:
            for term in set(doc_tokens):
                self._idf[term] = self._idf.get(term, 0.0) + 1.0
        total = max(self._doc_count, 1)
        for term in self._idf:
            self._idf[term] = (total / self._idf[term]) ** 0.5  # pseudo-IDF

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        """Return a score per document for *query_tokens*."""
        scores: list[float] = []
        for doc_tokens in self._corpus:
            score = 0.0
            for qt in query_tokens:
                tf = doc_tokens.count(qt) / max(len(doc_tokens), 1)
                score += tf * self._idf.get(qt, 0.5)
            scores.append(score)
        return scores


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

class BM25Retriever(BaseRetriever):
    """Sparse retriever using BM25 (or simple TF-IDF fallback).

    Parameters:
        documents: List of text strings to index.
        source: Source identifier applied to all returned RetrievedDoc
            entries (default ``"local_docs"``).
    """

    def __init__(self, documents: list[str], source: str = "local_docs") -> None:
        self._source = source
        self._raw_docs = documents
        self._tokenized: list[list[str]] = [_tokenize(doc) for doc in documents]

        if _HAS_RANK_BM25 and self._tokenized:
            self._bm25: Optional[BM25Okapi] = BM25Okapi(self._tokenized)
        else:
            self._bm25 = None

        self._fallback = _SimpleTFScorer(self._tokenized) if self._tokenized else None

        if not _HAS_RANK_BM25:
            logger.info("rank_bm25 not installed; using simple TF scorer")
        if not _HAS_JIEBA:
            logger.info("jieba not installed; using character-level tokenization")

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int) -> list[RetrievedDoc]:
        """Return up to *top_k* documents ranked by BM25."""
        if not self._raw_docs:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # Score documents
        if self._bm25 is not None:
            raw_scores = self._bm25.get_scores(query_tokens)
        elif self._fallback is not None:
            raw_scores = self._fallback.get_scores(query_tokens)
        else:
            return []

        # Build scored results
        scored: list[tuple[float, int]] = []
        for idx, score in enumerate(raw_scores):
            if score > 0:
                scored.append((float(score), idx))

        scored.sort(key=lambda x: -x[0])
        top = scored[:top_k]

        return [
            RetrievedDoc(
                content=self._raw_docs[idx],
                source=self._source,
                score=round(s, 4),
            )
            for s, idx in top
        ]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def document_count(self) -> int:
        """Return the number of indexed documents."""
        return len(self._raw_docs)
