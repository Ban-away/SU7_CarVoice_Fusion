"""Knowledge service facade.

Wires together local document storage, retrieval (local + optional web),
and citation synthesis into a single entry-point used by the orchestrator.
"""

from app.services.knowledge.local_store import LocalDocStore
from app.services.knowledge.models import RetrievedDoc  # noqa: F401 – re-export
from app.services.knowledge.retriever import Retriever
from app.services.knowledge.synthesizer import Synthesizer
from app.services.knowledge.web_search import WebSearchClient


class KnowledgeService:
    """Facade over the knowledge subsystem.

    Usage::

        ks = KnowledgeService(web_search_enabled=False)
        docs = ks.retrieve("SU7 续航", top_k=3)
        answer, citations = ks.synthesize_with_citations("SU7 续航", docs)
    """

    def __init__(
        self,
        web_search_enabled: bool = False,
        docs_path: str = "data/knowledge/su7_docs.json",
    ) -> None:
        self._local_store = LocalDocStore(docs_path=docs_path)
        self._web_client = WebSearchClient(enabled=web_search_enabled)
        self._retriever = Retriever(self._local_store, self._web_client)
        self._synthesizer = Synthesizer()

    # ------------------------------------------------------------------
    # Public API (backward-compatible with previous monolithic version)
    # ------------------------------------------------------------------

    def search_local_docs(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        """Search local document store only."""
        return self._local_store.search(query, top_k=top_k)

    def search_web_vertical(self, query: str) -> list[RetrievedDoc]:
        """Search web vertical (empty when disabled)."""
        return self._web_client.search(query)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        """Local-first retrieval with optional web fallback."""
        return self._retriever.retrieve(query, top_k=top_k)

    def synthesize_with_citations(
        self, query: str, docs: list[RetrievedDoc]
    ) -> tuple[str, list[dict]]:
        """Produce an answer string and structured citations from docs."""
        return self._synthesizer.synthesize(query, docs)
