"""Hybrid retriever combining BM25 (sparse) + FAISS (dense) with WRRF fusion.

Each sub-retriever independently searches the same document set, and
results are fused using Weighted Reciprocal Rank Fusion.
"""

import logging
import time

from app.knowledge.models import RetrievedDoc, RetrieveResult
from app.knowledge.retriever.base import BaseRetriever
from app.knowledge.retriever.bm25 import BM25Retriever
from app.knowledge.retriever.faiss import FAISSRetriever
from app.shared.utils import wrrf_fusion

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """Sparse + dense hybrid retriever with WRRF fusion.

    Parameters:
        documents: List of text strings shared by both sub-retrievers.
        source: Source identifier (default ``"local_docs"``).
        bm25_weight: Weight for BM25 results in WRRF (default 0.5).
        faiss_weight: Weight for FAISS results in WRRF (default 0.5).
        wrrf_k: Rank-decay constant for WRRF (default 60).
        embedding_dim: Dimensionality for random embedding fallback.
        model_name: Sentence-transformer model name for FAISS.
    """

    def __init__(
        self,
        documents: list[str],
        source: str = "local_docs",
        bm25_weight: float = 0.5,
        faiss_weight: float = 0.5,
        wrrf_k: int = 60,
        embedding_dim: int = 512,
        model_name: str = "BAAI/bge-small-zh-v1.5",
    ) -> None:
        self._bm25 = BM25Retriever(documents, source=source)
        self._faiss = FAISSRetriever(
            documents,
            source=source,
            model_name=model_name,
            embedding_dim=embedding_dim,
        )
        self._weights = [bm25_weight, faiss_weight]
        self._wrrf_k = wrrf_k

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int) -> list[RetrievedDoc]:
        """Return up to *top_k* documents after WRRF fusion."""
        result = self.retrieve_with_metadata(query, top_k)
        return result.docs

    # ------------------------------------------------------------------
    # Extended API
    # ------------------------------------------------------------------

    def retrieve_with_metadata(self, query: str, top_k: int) -> RetrieveResult:
        """Return a :class:`RetrieveResult` that includes latency_ms."""
        t0 = time.perf_counter()

        bm25_docs = self._bm25.retrieve(query, top_k=top_k * 2)
        faiss_docs = self._faiss.retrieve(query, top_k=top_k * 2)

        fused = wrrf_fusion(
            [bm25_docs, faiss_docs],
            weights=self._weights,
            k=self._wrrf_k,
        )
        fused = [doc for doc in fused if isinstance(doc, RetrievedDoc)]
        fused = fused[:top_k]

        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.debug(
            "Hybrid retrieve: bm25=%d, faiss=%d, fused=%d, latency=%dms",
            len(bm25_docs),
            len(faiss_docs),
            len(fused),
            latency_ms,
        )

        return RetrieveResult(docs=fused, latency_ms=latency_ms)
