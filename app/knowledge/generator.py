"""Answer generator — uses the LLM to produce answers from retrieved documents."""

import logging

from app.knowledge.models import RetrievedDoc
from app.llm.base import LLMMessage, create_llm_client
from app.shared.config import get_settings

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generate natural-language answers from a set of retrieved documents.

    The generator calls the configured LLM (via :func:`create_llm_client`)
    with a prompt that includes the user query and the top retrieved documents.

    Parameters:
        llm_provider: LLM provider override (if None, uses the configured
            default from settings).
        max_input_chars: Truncate each document to this many characters to
            keep the prompt within model context limits.
    """

    _SYSTEM_PROMPT = (
        "你是一个专业的车载语音助手知识引擎。"
        "请根据提供的参考文档片段回答用户问题。"
        "如果文档中不包含相关信息，请诚实地说'根据现有资料暂时无法回答'。"
        "回答要简洁、准确，适合语音播报。"
        "必要时使用【1】【2】标注引用来源。"
    )

    def __init__(
        self,
        llm_provider: str | None = None,
        max_input_chars: int = 1200,
    ) -> None:
        settings = get_settings()
        provider = llm_provider or settings.llm_provider
        self._llm = create_llm_client(provider)
        self._max_input_chars = max_input_chars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        docs: list[RetrievedDoc],
        *,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """Generate an answer from *docs* for *query*.

        Args:
            query: The user's question.
            docs: Retrieved documents to use as context.
            temperature: LLM sampling temperature.
            max_tokens: Maximum tokens in the generated response.

        Returns:
            The generated answer string.
        """
        if not docs:
            return "暂未检索到相关信息，请尝试更具体的问题。"

        context = self._format_context(docs)
        user_prompt = self._build_user_prompt(query, context)

        messages = [
            LLMMessage(role="system", content=self._SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        try:
            response = self._llm.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            logger.debug("Generated answer for query: %s", query[:50])
            return response.content.strip()
        except Exception:
            logger.exception("LLM generation failed; returning fallback")
            return self._fallback_answer(docs)

    def generate_stream(
        self,
        query: str,
        docs: list[RetrievedDoc],
        *,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ):
        """Generate an answer with streaming (yields text deltas)."""
        if not docs:
            yield "暂未检索到相关信息，请尝试更具体的问题。"
            return

        context = self._format_context(docs)
        user_prompt = self._build_user_prompt(query, context)

        messages = [
            LLMMessage(role="system", content=self._SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        try:
            for delta in self._llm.chat_stream(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yield delta
        except Exception:
            logger.exception("LLM streaming failed; returning fallback")
            yield self._fallback_answer(docs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_context(self, docs: list[RetrievedDoc]) -> str:
        """Format documents into a numbered context string."""
        parts: list[str] = []
        for i, doc in enumerate(docs, 1):
            text = doc.content[: self._max_input_chars]
            parts.append(f"【{i}】{text}")
        return "\n".join(parts)

    @staticmethod
    def _build_user_prompt(query: str, context: str) -> str:
        return f"请根据以下参考文档回答用户问题。\n\n参考文档：\n{context}\n\n用户问题：{query}\n\n回答："

    @staticmethod
    def _fallback_answer(docs: list[RetrievedDoc]) -> str:
        """Return a simple concatenation of the best documents as fallback."""
        top = docs[:2]
        return "；".join(doc.content for doc in top)
