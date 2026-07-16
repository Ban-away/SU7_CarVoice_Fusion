"""Knowledge service facade.

Wires together the full RAG pipeline: document loading, retrieval
(BM25 / FAISS / hybrid with WRRF fusion), optional reranking, LLM-based
answer generation with citation synthesis, and optional web search.

Provides the same public API as the legacy ``app.services.knowledge``
implementation for backward compatibility.
"""

import logging
from typing import Optional

from app.knowledge.models import RetrievedDoc, RetrieveResult
from app.knowledge.retriever.base import BaseRetriever
from app.knowledge.retriever.bm25 import BM25Retriever
from app.knowledge.retriever.faiss import FAISSRetriever
from app.knowledge.retriever.hybrid import HybridRetriever
from app.knowledge.reranker.base import BaseReranker
from app.knowledge.reranker.minicpm import MiniCPMReranker
from app.knowledge.synthesizer import CitationSynthesizer
from app.knowledge.web_search import WebSearchClient
from app.shared.config import get_settings

logger = logging.getLogger(__name__)


class KnowledgeService:
    """Facade over the knowledge subsystem.

    Parameters:
        web_search_enabled: Enable/disable web search.
        docs_path: Path to the JSON document file.
        retriever_backend: One of ``"bm25"``, ``"faiss"``, ``"hybrid"``,
            or ``"mock"``.  Default reads from config.
        reranker_backend: One of ``"minicpm"`` or ``"mock"``.
            Default reads from config.
        top_k: Default number of documents to retrieve.

    Usage::

        ks = KnowledgeService(web_search_enabled=False)
        docs = ks.retrieve("SU7 续航", top_k=3)
        answer, citations = ks.synthesize_with_citations("SU7 续航", docs)
    """

    def __init__(
        self,
        web_search_enabled: bool | None = None,
        docs_path: str | None = None,
        retriever_backend: str | None = None,
        reranker_backend: str | None = None,
        top_k: int | None = None,
    ) -> None:
        settings = get_settings()

        self._web_search_enabled = (
            web_search_enabled
            if web_search_enabled is not None
            else settings.web_search_enabled
        )
        self._docs_path = docs_path or settings.knowledge_docs_path
        self._retriever_backend = retriever_backend or settings.retriever_backend
        self._reranker_backend = reranker_backend or settings.reranker_backend
        self._top_k = top_k or settings.knowledge_top_k

        # ── Load documents ──
        self._documents: list[RetrievedDoc] = self._load_documents()

        # ── Retriever ──
        self._retriever: BaseRetriever = self._build_retriever()

        # ── Reranker ──
        self._reranker: Optional[BaseReranker] = self._build_reranker()

        # ── Sub-services ──
        self._web_client = WebSearchClient(enabled=self._web_search_enabled)
        self._synthesizer = CitationSynthesizer()

        logger.info(
            "KnowledgeService ready: retriever=%s, reranker=%s, web=%s, docs=%d",
            self._retriever_backend,
            self._reranker_backend,
            self._web_search_enabled,
            len(self._documents),
        )

    # ------------------------------------------------------------------
    # Public API (backward-compatible)
    # ------------------------------------------------------------------

    def search_local_docs(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedDoc]:
        """Search the local document store (no web fallback).

        Returns:
            List of RetrievedDoc sorted by descending relevance score.
        """
        k = top_k or self._top_k
        docs = self._retriever.retrieve(query, top_k=k)
        if self._reranker is not None:
            docs = self._reranker.rerank(query, docs)
        return docs[:k]

    def search_web_vertical(self, query: str) -> list[RetrievedDoc]:
        """Search the web vertical (empty list when disabled).

        Returns:
            List of RetrievedDoc from web search (may be empty).
        """
        return self._web_client.search(query)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedDoc]:
        """Local-first retrieval with optional web fallback.

        When local results are insufficient the retriever falls back to
        web vertical search (if enabled).

        Returns:
            List of RetrievedDoc sorted by descending relevance score.
        """
        k = top_k or self._top_k

        local_docs = self.search_local_docs(query, top_k=k)

        # If local already provides enough results, skip web
        if len(local_docs) >= max(1, k - 1):
            return local_docs

        web_docs = self._web_client.search(query)
        merged = local_docs + web_docs
        merged.sort(key=lambda d: d.score, reverse=True)
        return merged[:k]

    def synthesize_with_citations(
        self, query: str, docs: list[RetrievedDoc]
    ) -> tuple[str, list[dict]]:
        """Produce an answer string and structured citations from docs.

        Args:
            query: The user's question.
            docs: Retrieved documents.

        Returns:
            A tuple of ``(answer, citations)``.
        """
        return self._synthesizer.synthesize(query, docs)

    # ------------------------------------------------------------------
    # Extended API (new in this version)
    # ------------------------------------------------------------------

    def retrieve_with_metadata(
        self, query: str, top_k: int | None = None
    ) -> RetrieveResult:
        """Same as :meth:`retrieve` but returns :class:`RetrieveResult`
        which includes ``latency_ms``.

        Only supported when the underlying retriever provides timing
        (currently ``HybridRetriever``).
        """
        import time

        t0 = time.perf_counter()
        docs = self.retrieve(query, top_k=top_k)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return RetrieveResult(docs=docs, latency_ms=latency_ms)

    def generate_answer(
        self,
        query: str,
        docs: list[RetrievedDoc],
        *,
        use_llm: bool = True,
    ) -> str:
        """Generate a fluent natural-language answer from *docs*.

        Args:
            query: The user's question.
            docs: Retrieved documents.
            use_llm: When True, use the LLM-based AnswerGenerator;
                otherwise concatenate documents directly.

        Returns:
            The answer string.
        """
        if not use_llm or not docs:
            answer, _ = self._synthesizer.synthesize(query, docs)
            return answer

        from app.knowledge.generator import AnswerGenerator

        generator = AnswerGenerator()
        return generator.generate(query, docs)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _load_documents(self) -> list[RetrievedDoc]:
        """Load documents from the configured JSON path.

        Falls back to built-in default documents if the file is missing
        or cannot be parsed.
        """
        import json
        from pathlib import Path

        path = Path(self._docs_path)
        if not path.exists():
            logger.warning("Document file not found: %s; using defaults", path)
            return self._default_docs()

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read document file: %s; using defaults", exc)
            return self._default_docs()

        docs: list[RetrievedDoc] = []
        for item in raw:
            content = str(item.get("content", "")).strip()
            if content:
                docs.append(
                    RetrievedDoc(
                        content=content,
                        source=str(item.get("source", "local_docs")),
                        page=item.get("page"),
                    )
                )
        return docs or self._default_docs()

    @staticmethod
    def _default_docs() -> list[RetrievedDoc]:
        """Built-in fallback documents."""
        return [
            RetrievedDoc(
                content="小米 SU7 标准版 CLTC 续航约 700km，长续航版本可达更高里程。",
                source="su7_manual.pdf",
                page=12,
            ),
            RetrievedDoc(
                content="车机支持语音控制导航、媒体、空调和车辆设置，可通过唤醒词启动。",
                source="su7_quick_start.pdf",
                page=5,
            ),
            RetrievedDoc(
                content="车机可通过'你好小爱'语音唤醒。",
                source="su7_voice_guide.pdf",
                page=2,
            ),
        ]

    def _build_retriever(self) -> BaseRetriever:
        """Instantiate the configured retriever backend."""
        texts = [d.content for d in self._documents]

        if self._retriever_backend == "bm25":
            return BM25Retriever(texts)
        elif self._retriever_backend == "faiss":
            return FAISSRetriever(texts)
        elif self._retriever_backend == "hybrid":
            return HybridRetriever(texts)
        else:
            # mock: use a simple BM25 which degrades to TF scoring
            logger.info("Retriever backend '%s' -> using BM25 as fallback", self._retriever_backend)
            return BM25Retriever(texts)

    def _build_reranker(self) -> Optional[BaseReranker]:
        """Instantiate the configured reranker backend (or None)."""
        if self._reranker_backend == "minicpm":
            return MiniCPMReranker()
        # mock: no reranking
        logger.info("Reranker backend '%s' -> no reranking", self._reranker_backend)
        return None
