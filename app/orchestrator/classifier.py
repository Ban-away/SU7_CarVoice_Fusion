from dataclasses import dataclass


@dataclass
class ClassificationResult:
    route: str
    confidence: float


TASK_KEYWORDS = ["打开", "关闭", "播放", "导航", "前往", "去", "调大", "调小", "查询", "检查"]
FAQ_KEYWORDS = ["是什么", "怎么", "如何", "说明书", "手册", "续航", "参数", "支持", "充电"]
CHITCHAT_KEYWORDS = ["你好", "谢谢", "天气", "你是谁", "讲个笑话", "在吗"]


def classify_intent(message: str) -> ClassificationResult:
    lowered = message.strip().lower()
    if any(keyword in lowered for keyword in TASK_KEYWORDS):
        return ClassificationResult(route="Task", confidence=0.90)
    if any(keyword in lowered for keyword in FAQ_KEYWORDS):
        return ClassificationResult(route="FAQ", confidence=0.82)
    if any(keyword in lowered for keyword in CHITCHAT_KEYWORDS):
        return ClassificationResult(route="Chitchat", confidence=0.78)
    return ClassificationResult(route="Unknown", confidence=0.35)
