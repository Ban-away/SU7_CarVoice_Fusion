"""LLM client abstraction layer.

Provides a unified interface for different LLM backends
(Doubao, vLLM, OpenAI-compatible, and a local mock for development).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMMessage:
    role: str  # system | user | assistant
    content: str


@dataclass
class LLMResponse:
    content: str
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)


class BaseLLMClient(ABC):
    """Abstract LLM client."""

    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> LLMResponse:
        ...

    def chat_stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        """Yield text deltas. Override in streaming-capable clients."""
        resp = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        yield resp.content


def create_llm_client(provider: str = "mock", **kwargs) -> BaseLLMClient:
    """Factory: return the configured LLM client."""
    settings = None  # lazy import to avoid circular
    if provider == "doubao":
        from app.llm.doubao import DoubaoClient
        if not kwargs:
            from app.shared.config import get_settings
            s = get_settings()
            kwargs["api_key"] = s.doubao_api_key
            kwargs["base_url"] = s.doubao_endpoint
            kwargs["model_name"] = s.doubao_model
        return DoubaoClient(**kwargs)
    if provider == "vllm":
        from app.llm.vllm import VLLMClient
        return VLLMClient(**kwargs)
    if provider == "openai":
        from app.llm.openai_client import OpenAIClient
        return OpenAIClient(**kwargs)
    # default: mock
    from app.llm.mock import MockLLMClient
    return MockLLMClient()
