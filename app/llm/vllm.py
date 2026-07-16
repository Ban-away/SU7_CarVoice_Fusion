"""vLLM-compatible OpenAI client."""

import logging

from openai import OpenAI

from app.llm.base import BaseLLMClient, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class VLLMClient(BaseLLMClient):
    """Calls a vLLM server via its OpenAI-compatible endpoint."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000/v1", model: str = "qwen3-8b") -> None:
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
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
                stream=stream,
            )
            if stream:
                content = "".join(
                    chunk.choices[0].delta.content or ""
                    for chunk in resp
                    if chunk.choices[0].delta.content
                )
                return LLMResponse(content=content)
            choice = resp.choices[0]
            return LLMResponse(
                content=choice.message.content or "",
                finish_reason=choice.finish_reason or "stop",
                usage=resp.usage.model_dump() if resp.usage else {},
            )
        except Exception:
            logger.exception("vLLM call failed")
            return LLMResponse(content="")
