"""Main orchestrator — ties together classification, NLP, skills, and knowledge.

This is the central brain ported from CarVoice_Agent/dialog.py, rewritten
for the FastAPI fusion architecture.
"""

import time
import logging

from app.core.classifier import classify_intent
from app.core.session import SessionStore, session_store
from app.knowledge.service import KnowledgeService
from app.services.skills.registry import SkillsRegistry
from app.shared.config import get_settings
from app.shared.schemas import ChatResponse, Citation, RouteType, Trace

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """Orchestrates the full conversation pipeline.

    Pipeline:
    query → rewrite (multi-turn) → classify (rule + optional LLM arbitration)
    → route (Task/FAQ/Chitchat/Unknown) → execute/retrieve/chat/clarify
    → NLG (optional) → unified response
    """

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

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def handle(
        self,
        message: str,
        confirm: bool = False,
        session_id: str | None = None,
    ) -> ChatResponse:
        start_time = time.perf_counter()
        session = self.sessions.ensure(session_id)

        # ── Pending confirmation check ──
        if confirm:
            pending = self.sessions.consume_pending_confirmation(session.session_id)
            if pending:
                pending_skill, pending_message = pending
                result = self.skills_registry.execute(pending_skill, pending_message)
                # Optionally run NLG
                nlg_text = self._maybe_nlg(pending_message, result)
                self.sessions.append_history(session.session_id, message)
                return ChatResponse(
                    type="task_result",
                    text=nlg_text,
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

        # ── Query rewrite (multi-turn co-reference) ──
        rewritten = self._rewrite_with_context(message, session.session_id)

        # ── Classification ──
        classification = classify_intent(rewritten)

        # ── Route: Task ──
        if classification.route == "Task" and classification.confidence >= self.settings.task_confidence_threshold:
            return self._handle_task(rewritten, message, classification, session, start_time)

        # ── Route: FAQ ──
        if classification.route == "FAQ" and classification.confidence >= self.settings.faq_confidence_threshold:
            return self._handle_faq(rewritten, message, classification, session, start_time)

        # ── Route: Chitchat ──
        if classification.route == "Chitchat" and classification.confidence >= self.settings.chitchat_confidence_threshold:
            return self._handle_chitchat(rewritten, message, classification, session, start_time)

        # ── Route: Unknown / low confidence → clarification ──
        return self._clarification(
            route=classification.route,
            confidence=classification.confidence,
            fallback_reason="low_confidence",
            start_time=start_time,
            session_id=session.session_id,
            rewritten_query=rewritten,
        )

    # ------------------------------------------------------------------
    # Route handlers
    # ------------------------------------------------------------------

    def _handle_task(self, rewritten, message, classification, session, start_time):
        skill = self.skills_registry.resolve_skill(rewritten)
        if skill is None:
            # Try NLU extraction as fallback
            nlu_result = self._maybe_nlu(rewritten)
            if nlu_result and nlu_result.get("function") != "Unknown":
                # Route to DM handler
                dm_result = self._maybe_dm(nlu_result, rewritten)
                if dm_result:
                    nlg_text = self._maybe_nlg(rewritten, dm_result)
                    self.sessions.append_history(session.session_id, message)
                    return ChatResponse(
                        type="task_result", text=nlg_text, citations=[],
                        trace=Trace(
                            route="Task", classifier_confidence=classification.confidence,
                            latency_ms=self._latency_ms(start_time),
                            risk_level="medium", session_id=session.session_id,
                            rewritten_query=rewritten,
                        ),
                        session_id=session.session_id,
                    )
            return self._clarification(
                route="Task", confidence=classification.confidence,
                fallback_reason="skill_not_found", start_time=start_time,
                session_id=session.session_id, rewritten_query=rewritten,
            )

        # High-risk check
        if skill.risk_level == "high" and not self.sessions.consume_pending_confirmation(session.session_id):
            self.sessions.set_pending_confirmation(session.session_id, skill.name, rewritten)
            return self._clarification(
                route="Task", confidence=classification.confidence,
                fallback_reason="high_risk_needs_confirmation", start_time=start_time,
                text="该操作风险较高，请在同一会话中回复 confirm=true 后执行。",
                risk_level=skill.risk_level, session_id=session.session_id,
                rewritten_query=rewritten,
            )

        result = self.skills_registry.execute(skill.name, rewritten)
        nlg_text = self._maybe_nlg(rewritten, result)
        self.sessions.append_history(session.session_id, message)
        return ChatResponse(
            type="task_result", text=nlg_text, citations=[],
            trace=Trace(
                route="Task", classifier_confidence=classification.confidence,
                latency_ms=self._latency_ms(start_time),
                risk_level=skill.risk_level, session_id=session.session_id,
                rewritten_query=rewritten,
            ),
            session_id=session.session_id,
        )

    def _handle_faq(self, rewritten, message, classification, session, start_time):
        docs = self.knowledge_service.retrieve(rewritten, top_k=self.settings.knowledge_top_k)
        if not docs:
            return self._clarification(
                route="FAQ", confidence=classification.confidence,
                fallback_reason="insufficient_recall", start_time=start_time,
                session_id=session.session_id, rewritten_query=rewritten,
            )
        answer, citations = self.knowledge_service.synthesize_with_citations(rewritten, docs)
        self.sessions.append_history(session.session_id, message)
        return ChatResponse(
            type="faq_answer", text=answer,
            citations=[Citation(**item) for item in citations],
            trace=Trace(
                route="FAQ", classifier_confidence=classification.confidence,
                knowledge_hit_count=len(docs),
                latency_ms=self._latency_ms(start_time),
                session_id=session.session_id, rewritten_query=rewritten,
            ),
            session_id=session.session_id,
        )

    def _handle_chitchat(self, rewritten, message, classification, session, start_time):
        self.sessions.append_history(session.session_id, message)
        # Optionally use LLM chat
        chat_text = self._maybe_chat(rewritten, session.session_id)
        return ChatResponse(
            type="chitchat", text=chat_text, citations=[],
            trace=Trace(
                route="Chitchat", classifier_confidence=classification.confidence,
                latency_ms=self._latency_ms(start_time),
                session_id=session.session_id, rewritten_query=rewritten,
            ),
            session_id=session.session_id,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rewrite_with_context(self, message: str, session_id: str) -> str:
        """Multi-turn query rewrite — try LLM rewrite, fallback to simple concat."""
        previous = self.sessions.get_last_user_message(session_id)
        trimmed = message.strip()
        if not previous:
            return trimmed

        # Try LLM rewrite for short follow-ups
        if len(trimmed) <= 12:
            try:
                from app.nlp.rewrite import rewrite_query
                rewritten = rewrite_query(trimmed, session_id, last_answer="")
                if rewritten != trimmed:
                    logger.info("Query rewritten: %r → %r", trimmed, rewritten)
                    return rewritten
            except Exception:
                pass

        # Simple fallback
        if len(trimmed) <= 6 and any(t in previous for t in ["续航", "充电", "导航", "语音"]):
            return f"{previous}，补充问题：{trimmed}"
        return trimmed

    def _maybe_nlg(self, query: str, tool_response: str) -> str:
        """Optionally refine tool response via NLG."""
        try:
            from app.nlp.nlg import generate_nlg
            return generate_nlg(query, tool_response)
        except Exception:
            return tool_response

    def _maybe_chat(self, query: str, sender_id: str) -> str:
        """Optionally generate a chat response via LLM."""
        try:
            from app.llm.base import LLMMessage, create_llm_client
            from app.prompts.chat import BOT_CHAT_SYSTEM_PROMPT

            client = create_llm_client(self.settings.llm_provider)
            messages = [
                LLMMessage(role="system", content=BOT_CHAT_SYSTEM_PROMPT),
                LLMMessage(role="user", content=query),
            ]
            resp = client.chat(messages, max_tokens=100)
            return resp.content or "你好，我是 SU7 车载语音助手，很高兴为你服务。"
        except Exception:
            return "你好，我是 SU7 车载语音助手，很高兴为你服务。"

    def _maybe_nlu(self, query: str) -> dict | None:
        """Try NLU extraction."""
        try:
            from app.nlp.nlu import extract_intent
            return extract_intent(query)
        except Exception:
            return None

    def _maybe_dm(self, nlu_result: dict, query: str) -> str | None:
        """Try to execute via DM handler."""
        try:
            from app.skills.dm.factory import DMFactory
            func_name = nlu_result.get("function", "")
            # Map function to domain
            domain_map = {
                "Go_POI": "maps", "Search_Music": "music",
                "Query_Weather": "weather", "Query_Timely_Weather": "weather",
            }
            domain = domain_map.get(func_name, "")
            handler = DMFactory.get(domain)
            if handler:
                import asyncio
                _, text = asyncio.get_event_loop().run_until_complete(
                    handler(func_name, query, nlu_result.get("slots", {}))
                )
                return text
        except Exception:
            pass
        return None

    def _clarification(self, route, confidence, fallback_reason, start_time,
                       session_id, text=None, risk_level=None, rewritten_query=None):
        return ChatResponse(
            type="clarification",
            text=text or "我还不太确定你的意图，请补充你要执行的任务或问题细节。",
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

    @staticmethod
    def _latency_ms(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)

    @staticmethod
    def _normalize_route(route: str) -> RouteType:
        if route in ("Task", "FAQ", "Chitchat"):
            return route  # type: ignore[return-value]
        return "Unknown"
