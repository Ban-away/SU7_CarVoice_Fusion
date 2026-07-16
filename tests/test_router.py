from app.orchestrator.router import ChatOrchestrator


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
