"""Correlation — determines whether consecutive queries are semantically related.

Ported from CarVoice_Agent/client/correlation.py.
"""

import logging

from app.llm.base import LLMMessage, create_llm_client
from app.prompts.correlation import CORRELATION_PROMPT, CORRELATION_SYSTEM
from app.shared.config import get_settings
from app.shared.redis_client import get_redis

logger = logging.getLogger(__name__)


def check_correlation(
    query: str,
    sender_id: str,
    previous_query: str = "",
    was_rejected: bool = False,
) -> bool:
    """Return True if *query* is a continuation of the previous (rejected) query.

    Short-circuits:
    - Exact match → True
    - Previous was NOT rejected → False (correlation is irrelevant)
    - Otherwise → ask LLM
    """
    if not was_rejected:
        return False
    if previous_query and query.strip() == previous_query.strip():
        return True

    settings = get_settings()
    prompt = CORRELATION_PROMPT.format(previous_query, query)
    messages = [
        LLMMessage(role="system", content=CORRELATION_SYSTEM),
        LLMMessage(role="user", content=prompt),
    ]

    client = create_llm_client(settings.llm_provider)
    try:
        resp = client.chat(messages, temperature=0.0, max_tokens=5)
        return "是" in resp.content
    except Exception:
        logger.exception("Correlation LLM failed")
        return False
