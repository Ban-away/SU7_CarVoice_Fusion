"""Web search client with mock fallback.

Controlled by the ``web_search_enabled`` configuration flag.  When
disabled, returns an empty result set.  When enabled, returns curated
mock results keyed by topic — replace with a real search API in
production.
"""

import logging

from app.knowledge.models import RetrievedDoc
from app.shared.config import get_settings

logger = logging.getLogger(__name__)


class WebSearchClient:
    """Web vertical-search client with a built-in mock fallback.

    Parameters:
        enabled: Force enable/disable.  When ``None``, reads the
            ``WEB_SEARCH_ENABLED`` value from :func:`get_settings`.
    """

    # Curated mock responses keyed by topic keyword
    _MOCK_HINTS: dict[str, str] = {
        "续航": "官方发布中提到续航与环境温度、驾驶习惯和轮胎状态有关。",
        "充电": "官方建议优先使用快充网络并开启预约充电降低峰值电价。",
        "导航": "导航服务支持实时路况、充电站筛选和高速优选策略。",
        "语音": "语音助手支持连续对话、多指令识别和自定义唤醒词。",
        "空调": "空调系统支持分区温控、远程预调节和自动内外循环切换。",
        "安全": "SU7 全系标配 AEB、车道保持、盲区监测等主动安全功能。",
        "保养": "建议每 10,000 公里或 12 个月进行一次常规保养检查。",
        "胎压": "标准胎压为 2.5 bar，冬季可适当增加 0.1-0.2 bar。",
    }

    _FALLBACK = "垂直检索补充：建议参考小米汽车官方帮助中心获取最新信息。"

    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            settings = get_settings()
            self._enabled = settings.web_search_enabled
        else:
            self._enabled = enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[RetrievedDoc]:
        """Search the web vertical for *query*.

        Returns:
            A list of RetrievedDoc (empty when disabled or no match).
        """
        if not self._enabled:
            logger.debug("Web search disabled; returning empty result")
            return []

        result = self._mock_search(query)
        if result:
            logger.debug("Web search hit for query: %s", query[:60])
        else:
            logger.debug("Web search no match for query: %s", query[:60])
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @classmethod
    def _mock_search(cls, query: str) -> list[RetrievedDoc]:
        """Return curated mock results based on topic keywords in *query*."""
        best = cls._FALLBACK
        matched = False
        for key, text in cls._MOCK_HINTS.items():
            if key in query:
                best = text
                matched = True
                break

        if not matched:
            # Return the fallback but at a lower score
            return [
                RetrievedDoc(
                    content=best,
                    source="xiaomi_auto_web",
                    page=None,
                    score=0.5,
                )
            ]

        return [
            RetrievedDoc(
                content=best,
                source="xiaomi_auto_web",
                page=None,
                score=0.7,
            )
        ]

    @property
    def enabled(self) -> bool:
        """Whether web search is currently enabled."""
        return self._enabled
