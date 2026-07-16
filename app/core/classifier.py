"""Intent classifier — keyword-based with LLM arbitration fallback.

Unifies the rule-based classifier (from the MVP) with the LLM-driven
arbitration (from CarVoice_Agent).
"""

from dataclasses import dataclass

from app.shared.config import get_settings

# Keyword patterns (fast path — no LLM call needed)
TASK_KEYWORDS = [
    "打开", "关闭", "播放", "导航", "前往", "去", "调大", "调小", "查询", "检查",
    "空调", "车窗", "天窗", "座椅", "音量", "电话", "导航到", "播放音乐",
    "温度", "充电", "预约", "搜索", "切换", "收藏", "接听", "挂断",
]
FAQ_KEYWORDS = [
    "是什么", "怎么", "如何", "说明书", "手册", "续航", "参数", "支持",
    "充电", "故障", "操作", "方法", "什么意思", "为什么", "功能",
]
CHITCHAT_KEYWORDS = [
    "你好", "谢谢", "天气", "你是谁", "讲个笑话", "在吗", "诗", "故事",
    "翻译", "介绍", "推荐",
]


@dataclass
class ClassificationResult:
    route: str       # Task | FAQ | Chitchat | Unknown
    confidence: float


def classify_intent(message: str, use_llm: bool = True) -> ClassificationResult:
    """Classify *message* into one of four routes.

    Fast path: keyword matching.
    LLM path: when *use_llm* is True and the LLM provider is not 'mock',
              delegates to the full arbitration prompt.
    """
    settings = get_settings()
    lowered = message.strip().lower()

    # ── Fast path: keywords ──
    if any(kw in lowered for kw in TASK_KEYWORDS):
        return ClassificationResult(route="Task", confidence=0.90)

    if any(kw in lowered for kw in FAQ_KEYWORDS):
        return ClassificationResult(route="FAQ", confidence=0.82)

    if any(kw in lowered for kw in CHITCHAT_KEYWORDS):
        return ClassificationResult(route="Chitchat", confidence=0.78)

    # ── LLM path ──
    if use_llm and settings.llm_provider != "mock":
        try:
            from app.nlp.arbitration import arbitrate
            result = arbitrate(message)
            # Map arbitration routes to classifier routes
            route_map = {"task": "Task", "faq": "FAQ", "chat": "Chitchat", "unknown": "Unknown"}
            return ClassificationResult(
                route=route_map.get(result.route, "Unknown"),
                confidence=result.confidence,
            )
        except ImportError:
            pass

    return ClassificationResult(route="Unknown", confidence=0.35)
