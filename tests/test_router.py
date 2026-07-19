from app.core.orchestrator import ChatOrchestrator


def test_task_route_returns_task_result() -> None:
    orchestrator = ChatOrchestrator()
    response = orchestrator.handle("请导航到公司")

    assert response.type == "task_result"
    assert response.trace.route == "Task"


def test_faq_route_returns_citations() -> None:
    orchestrator = ChatOrchestrator()
    response = orchestrator.handle("SU7 续航是多少")

    assert response.type == "faq_answer"
    assert len(response.citations) > 0
    assert response.citations[0].source


def test_unknown_route_returns_clarification() -> None:
    orchestrator = ChatOrchestrator()
    response = orchestrator.handle("asdfghjkl")

    assert response.type == "clarification"
    assert response.trace.fallback_reason == "low_confidence"


def test_high_risk_skill_requires_confirmation() -> None:
    orchestrator = ChatOrchestrator()
    response = orchestrator.handle("请关闭安全系统")

    assert response.type == "clarification"
    assert response.trace.fallback_reason == "high_risk_needs_confirmation"


def test_pending_confirmation_executes_in_same_session() -> None:
    orchestrator = ChatOrchestrator()
    first = orchestrator.handle("请关闭安全系统")
    second = orchestrator.handle("确认", confirm=True, session_id=first.session_id)

    assert second.type == "task_result"
    assert second.trace.fallback_reason == "confirmed_pending_skill"


def test_high_risk_not_bypassed_without_confirm() -> None:
    """Bug fix: second call without confirm should NOT execute the high-risk skill."""
    orchestrator = ChatOrchestrator()
    first = orchestrator.handle("请关闭安全系统")
    assert first.type == "clarification"

    # Same message again WITHOUT confirm — should STILL ask for confirmation
    second = orchestrator.handle("请关闭安全系统", session_id=first.session_id)
    assert second.type == "clarification"
    assert second.trace.fallback_reason == "high_risk_needs_confirmation"


def test_task_skill_not_found_falls_back_to_chat() -> None:
    """When no skill matches and NLU fails, should fall back to chat (CarVoice original)."""
    orchestrator = ChatOrchestrator()
    response = orchestrator.handle("给我推荐一本书")
    # Not task/FAQ/chitchat → should fall to chat or clarification
    assert response.type in ("chitchat", "clarification")
