from dataclasses import dataclass


@dataclass
class RetrievedDoc:
    content: str
    source: str
    page: int | None = None
    score: float = 0.0


class KnowledgeService:
    def __init__(self, web_search_enabled: bool = False) -> None:
        self.web_search_enabled = web_search_enabled
        self._local_docs = [
            RetrievedDoc(content="小米 SU7 标准版 CLTC 续航 700km。", source="su7_manual.pdf", page=12),
            RetrievedDoc(content="小米 SU7 支持语音控制导航、媒体和空调。", source="su7_quick_start.pdf", page=5),
            RetrievedDoc(content="车机可通过‘你好小爱’语音唤醒。", source="su7_voice_guide.pdf", page=2),
        ]

    def search_local_docs(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        query_terms = [term for term in query.lower().split() if term]
        scored: list[RetrievedDoc] = []

        for doc in self._local_docs:
            score = 0.0
            content = doc.content.lower()
            for term in query_terms:
                if term in content:
                    score += 1.0
            if any(token in content for token in ["续航", "语音", "导航", "充电"]) and any(
                token in query for token in ["续航", "语音", "导航", "充电"]
            ):
                score += 1.0
            if score > 0:
                scored.append(RetrievedDoc(content=doc.content, source=doc.source, page=doc.page, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def search_web_vertical(self, query: str) -> list[RetrievedDoc]:
        if not self.web_search_enabled:
            return []
        return [
            RetrievedDoc(
                content=f"垂直检索补充：与“{query}”相关的信息建议参考官方公告。",
                source="xiaomi_auto_web",
                page=None,
                score=0.5,
            )
        ]

    def synthesize_with_citations(self, query: str, docs: list[RetrievedDoc]) -> tuple[str, list[dict]]:
        if not docs:
            return "暂未检索到足够信息，请补充更具体的问题。", []

        top_docs = docs[:2]
        answer = "；".join(doc.content for doc in top_docs)
        citations = [{"source": doc.source, "page": doc.page} for doc in top_docs]
        return answer, citations
