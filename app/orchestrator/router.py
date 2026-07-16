import time

from app.orchestrator.classifier import classify_intent
from app.services.knowledge.service import KnowledgeService
from app.services.skills.registry import SkillsRegistry
from app.shared.config import get_settings
from app.shared.schemas import ChatResponse, Citation, Trace


class ChatOrchestrator:
    def __init__(self, skills_registry: SkillsRegistry | None = None, knowledge_service: KnowledgeService | None = None) -> None:
        settings = get_settings()
        self.settings = settings
        self.skills_registry = skills_registry or SkillsRegistry()
        self.knowledge_service = knowledge_service or KnowledgeService(web_search_enabled=settings.web_search_enabled)

    def handle(self, message: str, confirm: bool = False) -> ChatResponse:
        start_time = time.perf_counter()
        classification = classify_intent(message)

        if classification.route == "Task" and classification.confidence >= self.settings.task_confidence_threshold:
            skill = self.skills_registry.resolve_skill(message)
            if skill is None:
                return self._clarification(
                    route="Task",
                    confidence=classification.confidence,
                    fallback_reason="skill_not_found",
                    start_time=start_time,
                )

            if skill.risk_level == "high" and not confirm:
                return self._clarification(
                    route="Task",
                    confidence=classification.confidence,
                    fallback_reason="high_risk_needs_confirmation",
                    start_time=start_time,
                    text="该操作风险较高，请回复确认后再执行。",
                    risk_level=skill.risk_level,
                )

            result = self.skills_registry.execute(skill.name, message)
            return ChatResponse(
                type="task_result",
                text=result,
                citations=[],
                trace=Trace(
                    route="Task",
                    classifier_confidence=classification.confidence,
                    latency_ms=self._latency_ms(start_time),
                    risk_level=skill.risk_level,
                ),
            )

        if classification.route == "FAQ" and classification.confidence >= self.settings.faq_confidence_threshold:
            docs = self.knowledge_service.search_local_docs(message, top_k=self.settings.knowledge_top_k)
            if not docs:
                docs = self.knowledge_service.search_web_vertical(message)

            if not docs:
                return self._clarification(
                    route="FAQ",
                    confidence=classification.confidence,
                    fallback_reason="insufficient_recall",
                    start_time=start_time,
                )

            answer, citations = self.knowledge_service.synthesize_with_citations(message, docs)
            return ChatResponse(
                type="faq_answer",
                text=answer,
                citations=[Citation(**item) for item in citations],
                trace=Trace(
                    route="FAQ",
                    classifier_confidence=classification.confidence,
                    knowledge_hit_count=len(docs),
                    latency_ms=self._latency_ms(start_time),
                ),
            )

        if classification.route == "Chitchat" and classification.confidence >= self.settings.chitchat_confidence_threshold:
            return ChatResponse(
                type="chitchat",
                text="你好，我是 SU7 车载语音助手，很高兴为你服务。",
                citations=[],
                trace=Trace(
                    route="Chitchat",
                    classifier_confidence=classification.confidence,
                    latency_ms=self._latency_ms(start_time),
                ),
            )

        return self._clarification(
            route=classification.route,
            confidence=classification.confidence,
            fallback_reason="low_confidence",
            start_time=start_time,
        )

    def _clarification(
        self,
        route: str,
        confidence: float,
        fallback_reason: str,
        start_time: float,
        text: str = "我还不太确定你的意图，请补充你要执行的任务或问题细节。",
        risk_level: str | None = None,
    ) -> ChatResponse:
        return ChatResponse(
            type="clarification",
            text=text,
            citations=[],
            trace=Trace(
                route=route,  # type: ignore[arg-type]
                classifier_confidence=confidence,
                latency_ms=self._latency_ms(start_time),
                fallback_reason=fallback_reason,
                risk_level=risk_level,
            ),
        )

    @staticmethod
    def _latency_ms(start_time: float) -> int:
        return int((time.perf_counter() - start_time) * 1000)
