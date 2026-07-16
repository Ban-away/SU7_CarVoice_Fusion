"""QA quality filter — removes low-quality or off-topic samples.

Ported from XIAOMI_SU7_RAG/src/gen_qa/qa_filter.py.
"""

import re
import logging
from app.data_pipeline.abbr_expander import expand_abbreviations

logger = logging.getLogger(__name__)

# Patterns that indicate low-quality input
_OFF_TOPIC_PATTERNS = [
    r"(诗|词|歌|曲|小说|故事|笑话|新闻|股票|天气|黄历|汇率)",
]

_LOW_QUALITY_QUESTION_PATTERNS = [
    r"^.{0,3}$",          # too short
    r"^.{200,}$",          # too long
    r"^(什么|怎么|如何|为什么|哪里|哪个|谁|多少)$",  # bare question word
]

_INVALID_ANSWER_PATTERNS = [
    r"(无答案|没有答案|无|不知道|不清楚|暂无|无相关信息)",
    r"^.{0,2}$",
]


def is_low_quality_question(question: str) -> bool:
    """Check if a question is too short, too long, or nonsensical."""
    q = question.strip()
    for pat in _LOW_QUALITY_QUESTION_PATTERNS:
        if re.match(pat, q):
            return True
    return False


def is_off_topic(text: str) -> bool:
    """Check if text contains off-topic content."""
    for pat in _OFF_TOPIC_PATTERNS:
        if re.search(pat, text):
            return True
    return False


def is_invalid_answer(answer: str) -> bool:
    """Check if an answer is empty, too short, or a 'no answer' placeholder."""
    a = answer.strip()
    for pat in _INVALID_ANSWER_PATTERNS:
        if re.search(pat, a):
            return True
    return False


def filter_qa_pairs(
    pairs: list[dict[str, str]],
    expand_abbr: bool = True,
) -> list[dict[str, str]]:
    """Filter and clean a list of QA pairs.

    Removes low-quality, off-topic, and invalid samples.
    Optionally expands abbreviations.
    """
    abbr_map = None
    if expand_abbr:
        from app.data_pipeline.abbr_expander import load_abbr_map
        abbr_map = load_abbr_map()

    clean: list[dict[str, str]] = []
    for pair in pairs:
        q = pair.get("question", "").strip()
        a = pair.get("answer", "").strip()

        if is_low_quality_question(q) or is_invalid_answer(a):
            continue
        if is_off_topic(q) and is_off_topic(a):
            continue

        if expand_abbr and abbr_map:
            q = expand_abbreviations(q, abbr_map)
            a = expand_abbreviations(a, abbr_map)

        clean.append({"question": q, "answer": a})

    logger.info("QA filter: %d → %d pairs", len(pairs), len(clean))
    return clean
