import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RetrievedDoc:
    content: str
    source: str
    page: int | None = None
    score: float = 0.0


class KnowledgeService:
    def __init__(self, web_search_enabled: bool = False, docs_path: str = "data/knowledge/su7_docs.json") -> None:
        self.web_search_enabled = web_search_enabled
        self.docs_path = docs_path
        self._local_docs = self._load_local_docs(docs_path)

    def _load_local_docs(self, docs_path: str) -> list[RetrievedDoc]:
        path = Path(docs_path)
        if not path.exists():
            return [
                RetrievedDoc(content="小米 SU7 标准版 CLTC 续航 700km。", source="su7_manual.pdf", page=12),
                RetrievedDoc(content="小米 SU7 支持语音控制导航、媒体和空调。", source="su7_quick_start.pdf", page=5),
                RetrievedDoc(content="车机可通过‘你好小爱’语音唤醒。", source="su7_voice_guide.pdf", page=2),
            ]

        raw = json.loads(path.read_text(encoding="utf-8"))
        docs: list[RetrievedDoc] = []
        for item in raw:
            docs.append(
                RetrievedDoc(
                    content=str(item.get("content", "")).strip(),
                    source=str(item.get("source", "local_docs")),
                    page=item.get("page"),
                )
            )
        return [doc for doc in docs if doc.content]

    def _score_doc(self, query: str, doc_content: str) -> float:
        normalized_query = query.lower().strip()
        normalized_doc = doc_content.lower()

        query_terms = [term for term in normalized_query.replace("，", " ").replace("。", " ").split() if term]
        keyword_score = sum(1.0 for term in query_terms if term in normalized_doc)

        overlap_chars = set(normalized_query) & set(normalized_doc)
        char_score = len([char for char in overlap_chars if char.strip()]) * 0.08

        topic_boost = 0.0
        for topic in ["续航", "导航", "语音", "充电", "空调", "胎压", "安全"]:
            if topic in normalized_query and topic in normalized_doc:
                topic_boost += 0.8

        return keyword_score + char_score + topic_boost

    def search_local_docs(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        scored: list[RetrievedDoc] = []
        for doc in self._local_docs:
            score = self._score_doc(query, doc.content)
            if score > 0.25:
                scored.append(RetrievedDoc(content=doc.content, source=doc.source, page=doc.page, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def search_web_vertical(self, query: str) -> list[RetrievedDoc]:
        if not self.web_search_enabled:
            return []

        hints = {
            "续航": "官方发布中提到续航与环境温度、驾驶习惯和轮胎状态有关。",
            "充电": "官方建议优先使用快充网络并开启预约充电降低峰值电价。",
            "导航": "导航服务支持实时路况、充电站筛选和高速优选策略。",
        }
        best = "垂直检索补充：建议参考小米汽车官方帮助中心获取最新信息。"
        for key, text in hints.items():
            if key in query:
                best = text
                break

        return [RetrievedDoc(content=best, source="xiaomi_auto_web", page=None, score=0.65)]

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        local_hits = self.search_local_docs(query, top_k=top_k)
        if len(local_hits) >= max(1, top_k - 1):
            return local_hits

        web_hits = self.search_web_vertical(query)
        merged = local_hits + web_hits
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged[:top_k]

    def synthesize_with_citations(self, query: str, docs: list[RetrievedDoc]) -> tuple[str, list[dict]]:
        if not docs:
            return "暂未检索到足够信息，请补充更具体的问题。", []

        top_docs = docs[:2]
        answer = "；".join(doc.content for doc in top_docs)
        citations = [{"source": doc.source, "page": doc.page} for doc in top_docs]
        return answer, citations
