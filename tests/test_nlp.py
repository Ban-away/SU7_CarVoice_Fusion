"""Tests for the NLP pipeline — arbitration, NLU, NLG, rewrite, reject, correlation."""

import pytest

from app.nlp.arbitration import arbitrate
from app.nlp.nlg import generate_nlg
from app.nlp.nlu import extract_intent
from app.nlp.reject import should_reject
from app.nlp.correlation import check_correlation


class TestArbitration:
    def test_arbitrate_task(self):
        result = arbitrate("打开空调")
        assert result.route in ("task", "faq", "chat", "unknown")

    def test_arbitrate_empty(self):
        result = arbitrate("")
        assert result.route in ("task", "faq", "chat", "unknown")

    def test_arbitration_result_has_confidence(self):
        result = arbitrate("播放周杰伦的歌")
        assert 0 <= result.confidence <= 1.0


class TestNLG:
    def test_nlg_returns_string(self):
        result = generate_nlg("播放音乐", "已播放周杰伦的晴天")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_nlg_empty_tool_response(self):
        result = generate_nlg("打开导航", "")
        assert isinstance(result, str)


class TestNLU:
    def test_nlu_fallback_no_url(self):
        result = extract_intent("打开空调")
        assert isinstance(result, dict)
        assert "function" in result
        assert "intent" in result

    def test_nlu_fallback_unknown(self):
        result = extract_intent("asdfghjkl")
        assert result["function"] == "Unknown"


class TestReject:
    def test_reject_no_service(self):
        assert should_reject("测试", trace_id="test") is False


class TestCorrelation:
    def test_correlation_not_rejected(self):
        assert check_correlation("query", "user1", "prev", was_rejected=False) is False

    def test_correlation_exact_match(self):
        assert check_correlation("same", "user1", "same", was_rejected=True) is True
