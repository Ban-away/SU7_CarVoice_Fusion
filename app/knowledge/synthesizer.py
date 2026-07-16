"""Citation synthesizer — formats retrieved documents into cited answers.

Provides a standalone class that can produce text answers with
structured citation metadata from a list of retrieval results.
"""

import logging
import re

from app.knowledge.models import RetrievedDoc

logger = logging.getLogger(__name__)


class CitationSynthesizer:
    """Assemble top retrieved documents into a cited-answer structure.

    Can operate in two modes:
    - **Simple**: direct concatenation with source/page metadata.
    - **LLM**: delegates to :class:`AnswerGenerator` for fluent NL generation.

    Parameters:
        use_llm: When True, use the LLM-backed AnswerGenerator; otherwise
            produce a simple concatenated answer.
        llm_provider: LLM provider override (optional, uses config default).
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_provider: str | None = None,
    ) -> None:
        self._use_llm = use_llm
        self._llm_provider = llm_provider
        self._generator = None  # lazily created

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        query: str,
        docs: list[RetrievedDoc],
    ) -> tuple[str, list[dict]]:
        """Produce (answer_text, citations_list) from *docs*.

        Args:
            query: The user's question.
            docs: Retrieved documents to synthesize from.

        Returns:
            A tuple of ``(answer, citations)`` where *citations* is a list
            of dicts with ``source``, ``page``, and ``content`` keys.
        """
        if not docs:
            return (
                "暂未检索到足够信息，请补充更具体的问题。",
                [],
            )

        if self._use_llm:
            answer = self._llm_synthesize(query, docs)
        else:
            answer = self._simple_synthesize(query, docs)

        citations = self._build_citations(docs)
        return answer, citations

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _simple_synthesize(
        self,
        query: str,  # noqa: ARG002  -- kept for interface consistency
        docs: list[RetrievedDoc],
    ) -> str:
        """Direct concatenation of top documents."""
        top_docs = docs[:3]
        return "；".join(doc.content for doc in top_docs)

    def _llm_synthesize(
        self,
        query: str,
        docs: list[RetrievedDoc],
    ) -> str:
        """Use the LLM generator for a fluent answer."""
        if self._generator is None:
            from app.knowledge.generator import AnswerGenerator

            self._generator = AnswerGenerator(llm_provider=self._llm_provider)

        return self._generator.generate(query, docs)

    @staticmethod
    def _build_citations(docs: list[RetrievedDoc]) -> list[dict]:
        """Build structured citation entries from the top documents."""
        citations: list[dict] = []
        for doc in docs[:3]:
            citations.append(
                {
                    "source": doc.source,
                    "page": doc.page,
                    "content": doc.content[:200],
                }
            )
        return citations

    @staticmethod
    def post_process(response: str, docs: list[RetrievedDoc]) -> dict:
        """Post-process an LLM answer: extract citation markers and related info.

        This is a static utility that can be called independently of the
        synthesize method, useful when answers come from external LLM calls.

        Returns:
            dict with keys: ``answer``, ``cite_pages``, ``related_images``.
        """
        all_cites = re.findall(r"[【](.*?)[】]", response)
        cites: list[int] = []
        for cite in all_cites:
            cite = re.sub(r"[{} 【】]", "", cite)
            cite = cite.replace(",", "，")
            cites.extend(int(k) for k in cite.split("，") if k.isdigit())
        cites = list(set(cites))

        answer = re.sub(r"[【](.*?)[】]", "", response)
        answer = re.sub(r"[{}【】]", "", answer)

        pages: list[int] = []
        related_images: list = []
        for index in cites:
            if index > len(docs) or index < 1:
                continue
            doc_ref = docs[index - 1]
            meta = (
                getattr(doc_ref, "metadata", {})
                if hasattr(doc_ref, "metadata")
                else {}
            )
            if isinstance(doc_ref, dict):
                meta = doc_ref.get("metadata", {})
            images = meta.get("images_info", [])
            page = meta.get("page")
            if page is not None:
                pages.append(page)
            for img in images:
                if img.get("title"):
                    related_images.append(img)

        pages = sorted(set(pages))
        if answer.strip() in ("无答案", "没有答案", "无", ""):
            pages = []
            related_images = []

        return {
            "answer": answer,
            "cite_pages": pages,
            "related_images": related_images,
        }
