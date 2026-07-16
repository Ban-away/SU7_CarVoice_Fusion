"""OpenAI-compatible LLM client."""

import logging

from openai import OpenAI

from app.llm.base import BaseLLMClient, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """Generic OpenAI-compatible API client."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
    ) -> None:
        self.client = OpenAI(api_key=api_key or "sk-placeholder", base_url=base_url)
        self.model = model

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> LLMResponse:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = resp.choices[0]
            return LLMResponse(
                content=choice.message.content or "",
                finish_reason=choice.finish_reason or "stop",
            )
        except Exception:
            logger.exception("OpenAI API call failed")
            return LLMResponse(content="")
