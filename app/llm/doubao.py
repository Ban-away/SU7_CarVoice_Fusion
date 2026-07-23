"""Doubao (ByteDance) LLM client.

Ported from CarVoice_Agent client modules.
"""

import json
import logging
import time

import httpx

from app.llm.base import BaseLLMClient, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class DoubaoClient(BaseLLMClient):
    """Calls the Doubao (Ark) LLM API endpoint."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        model_name: str = "ep-20240601170316-5dhwt",
    ) -> None:
        self.api_key = api_key
        self.endpoint = base_url.rstrip("/")
        self.model = model_name

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> LLMResponse:
        url = f"{self.endpoint}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            return LLMResponse(
                content=choice["message"]["content"],
                finish_reason=choice.get("finish_reason", "stop"),
                usage=data.get("usage", {}),
            )
        except Exception:
            logger.exception("Doubao API call failed")
            # Safe fallback
            return LLMResponse(content="A")

    def chat_stream(self, messages, *, temperature=0.7, max_tokens=512):
        url = f"{self.endpoint}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        try:
            with httpx.stream("POST", url, json=body, headers=headers, timeout=30.0) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0].get("delta", {}).get("content", "")
                        if delta:
                            yield delta
        except Exception:
            logger.exception("Doubao streaming failed")
            yield ""
