"""Unit tests for the intent classifier."""

import pytest

from app.orchestrator.classifier import ClassificationResult, classify_intent


@pytest.mark.parametrize(
    "message,expected_route,min_confidence",
    [
        ("请导航到公司", "Task", 0.85),
        ("播放音乐", "Task", 0.85),
        ("打开车窗", "Task", 0.85),
        ("SU7 续航是多少", "FAQ", 0.75),
        ("怎么充电", "FAQ", 0.75),
        ("说明书里怎么说的", "FAQ", 0.75),
        ("你好", "Chitchat", 0.70),
        ("谢谢", "Chitchat", 0.70),
        ("讲个笑话", "Chitchat", 0.70),
    ],
)
def test_classify_intent_known_routes(
    message: str, expected_route: str, min_confidence: float
) -> None:
    """High-confidence inputs should route to the expected category."""
    result = classify_intent(message)
    assert result.route == expected_route
    assert result.confidence >= min_confidence


def test_classify_intent_unknown() -> None:
    """Gibberish input should return Unknown with low confidence."""
    result = classify_intent("abcdefghijklmnop")
    assert result.route == "Unknown"
    assert result.confidence < 0.60


def test_classify_intent_task_keywords_have_highest_confidence() -> None:
    """Task keywords should yield the highest confidence tier."""
    task = classify_intent("打开空调")
    assert task.route == "Task"
    assert task.confidence >= 0.85


def test_classification_result_dataclass() -> None:
    """ClassificationResult should be a proper dataclass."""
    cr = ClassificationResult(route="FAQ", confidence=0.82)
    assert cr.route == "FAQ"
    assert cr.confidence == 0.82
