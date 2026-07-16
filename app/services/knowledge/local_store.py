"""Local document store: loading from disk and keyword-based retrieval."""

import json
from pathlib import Path

from app.services.knowledge.models import RetrievedDoc


class LocalDocStore:
    """In-memory document store with keyword-scoring retrieval.

    Loads documents from a JSON file at construction time.
    Falls back to a built-in default set when the file is missing.
    """

    def __init__(self, docs_path: str = "data/knowledge/su7_docs.json") -> None:
        self._docs = self._load(docs_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        """Return the top-k highest-scoring documents for *query*."""
        scored: list[RetrievedDoc] = []
        for doc in self._docs:
            score = self._score(query, doc.content)
            if score > 0.25:
                scored.append(
                    RetrievedDoc(
                        content=doc.content,
                        source=doc.source,
                        page=doc.page,
                        score=score,
                    )
                )
        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path_str: str) -> list[RetrievedDoc]:
        path = Path(path_str)
        if not path.exists():
            return LocalDocStore._default_docs()

        raw = json.loads(path.read_text(encoding="utf-8"))
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
        return docs

    @staticmethod
    def _default_docs() -> list[RetrievedDoc]:
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

    @staticmethod
    def _score(query: str, doc_content: str) -> float:
        """Simple keyword + character-overlap + topic-boost scorer."""
        normalized_query = query.lower().strip()
        normalized_doc = doc_content.lower()

        # Token-level keyword match
        query_terms = [
            t
            for t in normalized_query.replace("，", " ").replace("。", " ").split()
            if t
        ]
        keyword_score = sum(1.0 for t in query_terms if t in normalized_doc)

        # Character overlap bonus
        overlap = set(normalized_query) & set(normalized_doc)
        char_score = len([c for c in overlap if c.strip()]) * 0.08

        # Topic keyword boost
        topic_boost = 0.0
        for topic in ["续航", "导航", "语音", "充电", "空调", "胎压", "安全"]:
            if topic in normalized_query and topic in normalized_doc:
                topic_boost += 0.8

        return keyword_score + char_score + topic_boost
