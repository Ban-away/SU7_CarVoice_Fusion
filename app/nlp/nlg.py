"""NLG — converts structured tool responses to natural-language replies.

Ported from CarVoice_Agent/client/nlg.py.
"""

import logging

from app.llm.base import LLMMessage, create_llm_client
from app.prompts.nlg import DEFAULT_NLG, NLG_PROMPT
from app.shared.config import get_settings

logger = logging.getLogger(__name__)


def generate_nlg(query: str, tool_response: str) -> str:
    """Generate a friendly natural-language reply from a tool response.

    Args:
        query: The original user utterance.
        tool_response: Raw text returned by the skill handler.

    Returns:
        A concise, friendly Chinese reply.
    """
    settings = get_settings()
    prompt = NLG_PROMPT.format(query, tool_response)
    messages = [LLMMessage(role="user", content=prompt)]

    client = create_llm_client(settings.llm_provider)
    try:
        resp = client.chat(messages, temperature=0.3, max_tokens=128)
        return resp.content.strip() or DEFAULT_NLG
    except Exception:
        logger.exception("NLG LLM failed")
        return tool_response or DEFAULT_NLG
