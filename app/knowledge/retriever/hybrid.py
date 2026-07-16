"""Hybrid retriever — BM25 (sparse) + Milvus/FAISS (dense) with WRRF fusion.

Matches the original XIAOMI_SU7_RAG pipeline:
  BM25 (top-N) + Milvus (top-N) → WRRF fusion → top-K

When ``dense_backend="milvus"`` (default), uses BGE+SPLADE Milvus hybrid.
When ``dense_backend="faiss"``, uses sentence-transformers FAISS.
"""

from __future__ import annotations

import logging
import time

from app.knowledge.models import RetrievedDoc, RetrieveResult
from app.knowledge.retriever.base import BaseRetriever
from app.knowledge.retriever.bm25 import BM25Retriever
from app.shared.utils import wrrf_fusion

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """BM25 + dense (Milvus or FAISS) hybrid with WRRF fusion.

    Parameters:
        documents: Shared document texts.
        source: Source identifier.
        dense_backend: ``"milvus"`` (default, matching original SU7_RAG) or ``"faiss"``.
        bm25_weight: BM25 weight in WRRF (default 0.7, matching original).
        dense_weight: Dense weight in WRRF (default 0.7, matching original).
        wrrf_k: WRRF rank-decay constant (default 60, matching original).
    """

    def __init__(
        self,
        documents: list[str],
        source: str = "local_docs",
        dense_backend: str = "milvus",
        bm25_weight: float = 0.7,
        dense_weight: float = 0.7,
        wrrf_k: int = 60,
    ) -> None:
        self._bm25 = BM25Retriever(documents, source=source)

        if dense_backend == "milvus":
            from app.knowledge.retriever.milvus import MilvusRetriever
            self._dense: BaseRetriever = MilvusRetriever(documents, source=source)
        elif dense_backend == "faiss":
            from app.knowledge.retriever.faiss import FAISSRetriever
            self._dense = FAISSRetriever(documents, source=source)
        else:
            raise ValueError(f"Unknown dense_backend: {dense_backend}")

        self._weights = [bm25_weight, dense_weight]
        self._wrrf_k = wrrf_k
        self._dense_backend = dense_backend

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int) -> list[RetrievedDoc]:
        result = self.retrieve_with_metadata(query, top_k)
        return result.docs

    def retrieve_with_metadata(self, query: str, top_k: int) -> RetrieveResult:
        """BM25 + dense → WRRF fusion, matching original SU7_RAG flow."""
        t0 = time.perf_counter()

        # BM25: top-k * 4 for broader recall (original used top-15 vs top-40 ratio)
        bm25_docs = self._bm25.retrieve(query, top_k=max(top_k, 15))
        dense_docs = self._dense.retrieve(query, top_k=max(top_k * 3, 40))

        fused = wrrf_fusion(
            [bm25_docs, dense_docs],
            weights=self._weights,
            k=self._wrrf_k,
        )
        fused = [d for d in fused if isinstance(d, RetrievedDoc)]
        fused = fused[:top_k]

        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.debug(
            "Hybrid(%s): bm25=%d, dense=%d, fused=%d, latency=%dms",
            self._dense_backend,
            len(bm25_docs),
            len(dense_docs),
            len(fused),
            latency_ms,
        )

        return RetrieveResult(docs=fused, latency_ms=latency_ms)
