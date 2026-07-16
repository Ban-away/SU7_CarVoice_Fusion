"""LLM-driven intent arbitration: classifies queries as Task / FAQ / Chitchat / Noise.

Ported from CarVoice_Agent/client/arbitration.py.
"""

import logging
from dataclasses import dataclass

from app.llm.base import LLMMessage, create_llm_client
from app.prompts.arbitration import ARBITRATION_SYSTEM_PROMPT
from app.shared.config import get_settings

logger = logging.getLogger(__name__)

# A=Task, B=FAQ, C=Chitchat, D=Noise
_CLASS_MAP = {"A": "task", "B": "faq", "C": "chat", "D": "unknown"}


@dataclass
class ArbitrationResult:
    route: str  # task | faq | chat | unknown
    confidence: float
    raw_label: str  # A | B | C | D


def arbitrate(
    query: str,
    history: list[str] | None = None,
) -> ArbitrationResult:
    """Classify *query* into one of four routes using the LLM arbitration prompt.

    Args:
        query: Current user utterance.
        history: Optional prior conversation turns (user messages only).

    Returns:
        ArbitrationResult with route, confidence, and raw LLM label.
    """
    settings = get_settings()

    # Build messages
    messages = [LLMMessage(role="system", content=ARBITRATION_SYSTEM_PROMPT)]
    if history:
        for h in history[-6:]:  # max 6-turn history
            messages.append(LLMMessage(role="user", content=h))
    messages.append(LLMMessage(role="user", content=query))

    client = create_llm_client(settings.llm_provider)
    try:
        resp = client.chat(messages, temperature=0.0, max_tokens=5)
        raw = resp.content.strip().upper()
    except Exception:
        logger.exception("Arbitration LLM failed, defaulting to task")
        raw = "A"

    # Extract label
    label = raw[0] if raw and raw[0] in "ABCD" else "A"
    route = _CLASS_MAP.get(label, "task")

    # Confidence heuristic: LLM → high; fallback → lower
    confidence = 0.92 if label != "A" or raw == "A" else 0.90
    # Adjust per route
    if route == "faq":
        confidence = 0.85
    elif route == "chat":
        confidence = 0.80
    elif route == "unknown":
        confidence = 0.90  # LLM is confident about noise

    return ArbitrationResult(route=route, confidence=confidence, raw_label=label)
