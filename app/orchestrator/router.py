import time

from app.orchestrator.classifier import classify_intent
from app.orchestrator.session import SessionStore, session_store
from app.services.knowledge.service import KnowledgeService
from app.services.skills.registry import SkillsRegistry
from app.shared.config import get_settings
from app.shared.schemas import ChatResponse, Citation, RouteType, Trace


class ChatOrchestrator:
    def __init__(
        self,
        skills_registry: SkillsRegistry | None = None,
        knowledge_service: KnowledgeService | None = None,
        sessions: SessionStore | None = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.skills_registry = skills_registry or SkillsRegistry()
        self.knowledge_service = knowledge_service or KnowledgeService(
            web_search_enabled=settings.web_search_enabled,
            docs_path=settings.knowledge_docs_path,
        )
        self.sessions = sessions or session_store

    def handle(self, message: str, confirm: bool = False, session_id: str | None = None) -> ChatResponse:
        start_time = time.perf_counter()
        session = self.sessions.ensure(session_id)

        if confirm:
            pending = self.sessions.consume_pending_confirmation(session.session_id)
            if pending:
                pending_skill, pending_message = pending
                result = self.skills_registry.execute(pending_skill, pending_message)
                self.sessions.append_history(session.session_id, message)
                return ChatResponse(
                    type="task_result",
                    text=result,
                    citations=[],
                    trace=Trace(
                        route="Task",
                        classifier_confidence=1.0,
                        latency_ms=self._latency_ms(start_time),
                        risk_level="high",
                        fallback_reason="confirmed_pending_skill",
                        session_id=session.session_id,
                    ),
                    session_id=session.session_id,
                )

        rewritten_message = self._rewrite_with_context(message, session.session_id)
        classification = classify_intent(rewritten_message)

        if classification.route == "Task" and classification.confidence >= self.settings.task_confidence_threshold:
            skill = self.skills_registry.resolve_skill(rewritten_message)
            if skill is None:
                return self._clarification(
                    route="Task",
                    confidence=classification.confidence,
                    fallback_reason="skill_not_found",
                    start_time=start_time,
                    session_id=session.session_id,
                    rewritten_query=rewritten_message,
                )

            if skill.risk_level == "high" and not confirm:
                self.sessions.set_pending_confirmation(session.session_id, skill.name, rewritten_message)
                return self._clarification(
                    route="Task",
                    confidence=classification.confidence,
                    fallback_reason="high_risk_needs_confirmation",
                    start_time=start_time,
                    text="该操作风险较高，请在同一会话中回复 confirm=true 后执行。",
                    risk_level=skill.risk_level,
                    session_id=session.session_id,
                    rewritten_query=rewritten_message,
                )

            result = self.skills_registry.execute(skill.name, rewritten_message)
            self.sessions.append_history(session.session_id, message)
            return ChatResponse(
                type="task_result",
                text=result,
                citations=[],
                trace=Trace(
                    route="Task",
                    classifier_confidence=classification.confidence,
                    latency_ms=self._latency_ms(start_time),
                    risk_level=skill.risk_level,
                    session_id=session.session_id,
                    rewritten_query=rewritten_message,
                ),
                session_id=session.session_id,
            )

        if classification.route == "FAQ" and classification.confidence >= self.settings.faq_confidence_threshold:
            docs = self.knowledge_service.retrieve(rewritten_message, top_k=self.settings.knowledge_top_k)

            if not docs:
                return self._clarification(
                    route="FAQ",
                    confidence=classification.confidence,
                    fallback_reason="insufficient_recall",
                    start_time=start_time,
                    session_id=session.session_id,
                    rewritten_query=rewritten_message,
                )

            answer, citations = self.knowledge_service.synthesize_with_citations(rewritten_message, docs)
            self.sessions.append_history(session.session_id, message)
            return ChatResponse(
                type="faq_answer",
                text=answer,
                citations=[Citation(**item) for item in citations],
                trace=Trace(
                    route="FAQ",
                    classifier_confidence=classification.confidence,
                    knowledge_hit_count=len(docs),
                    latency_ms=self._latency_ms(start_time),
                    session_id=session.session_id,
                    rewritten_query=rewritten_message,
                ),
                session_id=session.session_id,
            )

        if classification.route == "Chitchat" and classification.confidence >= self.settings.chitchat_confidence_threshold:
            self.sessions.append_history(session.session_id, message)
            return ChatResponse(
                type="chitchat",
                text="你好，我是 SU7 车载语音助手，很高兴为你服务。",
                citations=[],
                trace=Trace(
                    route="Chitchat",
                    classifier_confidence=classification.confidence,
                    latency_ms=self._latency_ms(start_time),
                    session_id=session.session_id,
                    rewritten_query=rewritten_message,
                ),
                session_id=session.session_id,
            )

        return self._clarification(
            route=classification.route,
            confidence=classification.confidence,
            fallback_reason="low_confidence",
            start_time=start_time,
            session_id=session.session_id,
            rewritten_query=rewritten_message,
        )

    def _clarification(
        self,
        route: str,
        confidence: float,
        fallback_reason: str,
        start_time: float,
        session_id: str,
        text: str = "我还不太确定你的意图，请补充你要执行的任务或问题细节。",
        risk_level: str | None = None,
        rewritten_query: str | None = None,
    ) -> ChatResponse:
        return ChatResponse(
            type="clarification",
            text=text,
            citations=[],
            trace=Trace(
                route=self._normalize_route(route),
                classifier_confidence=confidence,
                latency_ms=self._latency_ms(start_time),
                fallback_reason=fallback_reason,
                risk_level=risk_level,
                session_id=session_id,
                rewritten_query=rewritten_query,
            ),
            session_id=session_id,
        )

    def _rewrite_with_context(self, message: str, session_id: str) -> str:
        previous = self.sessions.get_last_user_message(session_id)
        trimmed = message.strip()
        if not previous:
            return trimmed
        if len(trimmed) <= 6 and any(token in previous for token in ["续航", "充电", "导航", "语音"]):
            return f"{previous}，补充问题：{trimmed}"
        return trimmed

    @staticmethod
    def _latency_ms(start_time: float) -> int:
        return int((time.perf_counter() - start_time) * 1000)

    @staticmethod
    def _normalize_route(route: str) -> RouteType:
        if route == "Task":
            return "Task"
        if route == "FAQ":
            return "FAQ"
        if route == "Chitchat":
            return "Chitchat"
        return "Unknown"
