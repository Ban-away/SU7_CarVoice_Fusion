"""Optional web vertical-search client.

Disabled by default (config-driven).  Provides a mock implementation
that returns curated hints keyed by topic.
"""

from app.services.knowledge.models import RetrievedDoc


class WebSearchClient:
    """Web vertical-search with a built-in mock fallback.

    Set *enabled* to ``True`` via configuration to activate the
    mock web layer; replace with a real search backend in production.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def search(self, query: str) -> list[RetrievedDoc]:
        """Return web-search hits for *query*, or an empty list when disabled."""
        if not self.enabled:
            return []

        # Curated mock responses keyed by topic
        hints: dict[str, str] = {
            "续航": "官方发布中提到续航与环境温度、驾驶习惯和轮胎状态有关。",
            "充电": "官方建议优先使用快充网络并开启预约充电降低峰值电价。",
            "导航": "导航服务支持实时路况、充电站筛选和高速优选策略。",
        }
        fallback = "垂直检索补充：建议参考小米汽车官方帮助中心获取最新信息。"

        best = fallback
        for key, text in hints.items():
            if key in query:
                best = text
                break

        return [
            RetrievedDoc(
                content=best,
                source="xiaomi_auto_web",
                page=None,
                score=0.65,
            )
        ]
