"""Citation synthesis: converts retrieved documents into an answer
with structured citations (source + page).
"""

from app.services.knowledge.models import RetrievedDoc


class Synthesizer:
    """Assembles top retrieved documents into a cited answer."""

    def synthesize(self, query: str, docs: list[RetrievedDoc]) -> tuple[str, list[dict]]:
        """Return (answer_text, citations_list) for the given docs.

        Citations include ``source`` (required) and ``page`` (nullable).
        """
        if not docs:
            return "暂未检索到足够信息，请补充更具体的问题。", []

        top_docs = docs[:2]
        answer = "；".join(doc.content for doc in top_docs)
        citations = [{"source": doc.source, "page": doc.page} for doc in top_docs]
        return answer, citations
