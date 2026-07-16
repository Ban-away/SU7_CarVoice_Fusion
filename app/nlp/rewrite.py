"""Query rewrite — co-reference resolution for multi-turn dialog.

Ported from CarVoice_Agent/client/rewrite.py.
"""

import logging

from app.llm.base import LLMMessage, create_llm_client
from app.prompts.rewrite import REWRITE_SYSTEM_PROMPT
from app.shared.config import get_settings
from app.shared.redis_client import get_redis

logger = logging.getLogger(__name__)

_HISTORY_TTL = 40   # seconds
_MAX_TURNS = 6


def rewrite_query(
    query: str,
    sender_id: str,
    last_answer: str = "",
) -> str:
    """Resolve pronouns, anaphora, and incomplete utterances.

    Returns the rewritten query, or the original if no rewrite is needed.
    """
    settings = get_settings()
    redis = get_redis()
    history_key = f"voice:rewrite_history:{sender_id}"

    # Load history
    raw = redis.get(history_key)
    history: list[str] = raw if isinstance(raw, list) else []
    if not history:
        return query

    # Format as A/B speaker turns
    turns: list[str] = []
    for i, msg in enumerate(history):
        role = "A" if i % 2 == 0 else "B"
        turns.append(f"{role}: {msg}")

    prompt = REWRITE_SYSTEM_PROMPT + "\n" + "\n".join(turns) + f"\nA: {query}"
    messages = [LLMMessage(role="user", content=prompt)]

    client = create_llm_client(settings.llm_provider)
    try:
        resp = client.chat(messages, temperature=0.0, max_tokens=128)
        rewritten = resp.content.strip()
    except Exception:
        logger.exception("Rewrite LLM failed")
        rewritten = query

    # Safeguard: discard if less than 25% character overlap
    if rewritten and _char_overlap(query, rewritten) >= 0.25:
        result = rewritten
    else:
        result = query

    # Update history
    history.append(query)
    if len(history) > _MAX_TURNS * 2:
        history = history[-(_MAX_TURNS * 2):]
    redis.set(history_key, history, ex=_HISTORY_TTL)

    return result


def _char_overlap(a: str, b: str) -> float:
    set_a = set(a)
    if not set_a:
        return 1.0
    set_b = set(b)
    return len(set_a & set_b) / len(set_a)
