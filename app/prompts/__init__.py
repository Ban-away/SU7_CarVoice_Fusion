"""Centralised prompt registry — every LLM prompt used across the system.

Ported verbatim from CarVoice_Agent/prompts.py.
"""

from app.prompts.arbitration import ARBITRATION_SYSTEM_PROMPT  # noqa: F401
from app.prompts.chat import BOT_CHAT_SYSTEM_PROMPT  # noqa: F401
from app.prompts.correlation import CORRELATION_PROMPT, CORRELATION_SYSTEM  # noqa: F401
from app.prompts.nlg import DEFAULT_NLG, NLG_PROMPT  # noqa: F401
from app.prompts.nlu import NLU_SYSTEM_PROMPT  # noqa: F401
from app.prompts.rewrite import REWRITE_SYSTEM_PROMPT  # noqa: F401
