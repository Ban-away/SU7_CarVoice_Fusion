"""LLM-driven QA pair generation from document chunks.

Ported from XIAOMI_SU7_RAG/src/gen_qa/run.py.
"""

import logging

from app.llm.base import LLMMessage, create_llm_client
from app.shared.config import get_settings

logger = logging.getLogger(__name__)

CONTEXT_PROMPT_TPL = """你是一个专业的汽车手册问答对生成专家。请根据以下文档片段，生成高质量的问答对。

要求：
1. 问题要具体、有实际意义，模拟真实车主的提问方式
2. 答案要准确、简洁，直接基于文档内容回答
3. 生成3-5个不重复的问答对
4. 输出格式：Q: ...\\nA: ...

文档片段：
{context}

请生成问答对："""

GENERALIZE_PROMPT_TPL = """请为以下问题生成5个语义相同但表述不同的变体问题，模拟不同车主可能的提问方式。

原问题：{question}

输出格式（每行一个变体）："""


def generate_qa_pairs(documents: list[str]) -> list[dict[str, str]]:
    """Generate QA pairs from a list of document content strings.

    Returns a list of {"question": ..., "answer": ...} dicts.
    """
    settings = get_settings()
    client = create_llm_client(settings.llm_provider)
    qa_pairs: list[dict[str, str]] = []

    for doc in documents:
        prompt = CONTEXT_PROMPT_TPL.format(context=doc[:2000])
        messages = [LLMMessage(role="user", content=prompt)]
        try:
            resp = client.chat(messages, temperature=0.7, max_tokens=1024)
            pairs = _parse_qa_output(resp.content)
            qa_pairs.extend(pairs)
        except Exception:
            logger.exception("QA generation failed for chunk")

    return qa_pairs


def generate_paraphrases(question: str, count: int = 5) -> list[str]:
    """Generate synonymous paraphrases of a question."""
    settings = get_settings()
    client = create_llm_client(settings.llm_provider)
    prompt = GENERALIZE_PROMPT_TPL.format(question=question)
    messages = [LLMMessage(role="user", content=prompt)]

    try:
        resp = client.chat(messages, temperature=0.8, max_tokens=512)
        return [
            line.strip().lstrip("0123456789.、) ").strip()
            for line in resp.content.strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
        ][:count]
    except Exception:
        logger.exception("Paraphrase generation failed")
        return [question]


def _parse_qa_output(text: str) -> list[dict[str, str]]:
    """Parse LLM QA output into structured pairs."""
    pairs: list[dict[str, str]] = []
    lines = text.strip().split("\n")
    current_q = ""
    for line in lines:
        line = line.strip()
        if line.startswith("Q:") or line.startswith("Q：") or line.startswith("问："):
            current_q = line.split(":", 1)[-1].split("：", 1)[-1].strip()
        elif (line.startswith("A:") or line.startswith("A：") or line.startswith("答：")) and current_q:
            answer = line.split(":", 1)[-1].split("：", 1)[-1].strip()
            pairs.append({"question": current_q, "answer": answer})
            current_q = ""
    return pairs
