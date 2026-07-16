from dataclasses import dataclass
from functools import lru_cache
import os


def _to_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "SU7 CarVoice Fusion")
    app_env: str = os.getenv("APP_ENV", "dev")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    task_confidence_threshold: float = float(os.getenv("TASK_CONFIDENCE_THRESHOLD", "0.75"))
    faq_confidence_threshold: float = float(os.getenv("FAQ_CONFIDENCE_THRESHOLD", "0.65"))
    chitchat_confidence_threshold: float = float(os.getenv("CHITCHAT_CONFIDENCE_THRESHOLD", "0.60"))

    knowledge_top_k: int = int(os.getenv("KNOWLEDGE_TOP_K", "3"))
    web_search_enabled: bool = _to_bool(os.getenv("WEB_SEARCH_ENABLED"), False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
