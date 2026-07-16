"""Mock LLM client — deterministic responses for development and testing."""

from app.llm.base import BaseLLMClient, LLMMessage, LLMResponse


class MockLLMClient(BaseLLMClient):
    """Returns rule-based responses, no external API needed.

    Used when LLM_PROVIDER=mock (default).
    """

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> LLMResponse:
        last = messages[-1].content if messages else ""

        # ── Arbitration simulation ──
        if any(kw in last for kw in ["意图识别", "A、B、C、D"]):
            return self._arbitrate(last)

        # ── NLG simulation ──
        if "工具返回" in last:
            return self._nlg(last)

        # ── Rewrite simulation ──
        if "改写" in last or "A:" in last:
            return LLMResponse(content=last.split("A:")[-1].strip().split("\n")[0] if "A:" in last else last)

        # ── Correlation simulation ──
        if "相关性" in last or "句子1" in last:
            return LLMResponse(content="是")

        # ── Chat simulation ──
        return LLMResponse(content="你好，我是 SU7 车载语音助手，很高兴为你服务。")

    def _arbitrate(self, text: str) -> LLMResponse:
        """Rule-based arbitration: A=Task, B=FAQ, C=Chitchat, D=Noise."""
        task_kw = ["打开", "关闭", "播放", "导航", "前往", "去", "调大", "调小", "查询", "检查",
                    "空调", "车窗", "天窗", "座椅", "音量", "电话", "导航到", "播放音乐"]
        faq_kw = ["怎么", "如何", "说明书", "手册", "续航", "参数", "支持", "充电",
                   "是什么", "故障", "操作", "方法"]
        chat_kw = ["你好", "谢谢", "天气", "你是谁", "讲个笑话", "在吗", "诗"]

        if any(kw in text for kw in task_kw):
            return LLMResponse(content="A")
        if any(kw in text for kw in faq_kw):
            return LLMResponse(content="B")
        if any(kw in text for kw in chat_kw):
            return LLMResponse(content="C")
        return LLMResponse(content="D")

    def _nlg(self, text: str) -> LLMResponse:
        """Simple NLG: extract tool response and wrap it."""
        if "工具返回：" in text:
            tool_resp = text.split("工具返回：", 1)[-1].strip()
            return LLMResponse(content=f"好的，{tool_resp}")
        return LLMResponse(content="好的，已执行。")
