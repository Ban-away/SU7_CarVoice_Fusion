"""Retriever: local-first with optional web fallback."""

from app.services.knowledge.local_store import LocalDocStore
from app.services.knowledge.models import RetrievedDoc
from app.services.knowledge.web_search import WebSearchClient


class Retriever:
    """Orchestrates local and web search, preferring local results.

    When local hits are insufficient the retriever falls back to
    web vertical search (if enabled).
    """

    def __init__(self, local_store: LocalDocStore, web_client: WebSearchClient) -> None:
        self._local = local_store
        self._web = web_client

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        """Return up to *top_k* documents, local-first with web fallback."""
        local_hits = self._local.search(query, top_k=top_k)

        # If local already provides enough results, skip web
        if len(local_hits) >= max(1, top_k - 1):
            return local_hits

        web_hits = self._web.search(query)
        merged = local_hits + web_hits
        merged.sort(key=lambda d: d.score, reverse=True)
        return merged[:top_k]
