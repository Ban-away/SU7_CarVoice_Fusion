"""Tests for the LLM abstraction layer."""

from app.llm.base import LLMMessage, create_llm_client
from app.llm.mock import MockLLMClient


class TestMockLLM:
    def test_mock_chat(self):
        client = MockLLMClient()
        response = client.chat([LLMMessage(role="user", content="你好")])
        assert response.content
        assert len(response.content) > 0

    def test_mock_arbitration(self):
        client = MockLLMClient()
        response = client.chat([LLMMessage(role="user", content="意图识别 A、B、C、D")])
        assert response.content in "ABCD"


class TestFactory:
    def test_create_mock_client(self):
        client = create_llm_client("mock")
        assert isinstance(client, MockLLMClient)

    def test_create_unknown_falls_back_to_mock(self):
        client = create_llm_client("nonexistent_provider")
        response = client.chat([LLMMessage(role="user", content="test")])
        assert response.content
