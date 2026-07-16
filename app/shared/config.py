"""Application configuration — environment-driven settings.

Expanded for the full CarVoice_Agent + XIAOMI_SU7_RAG fusion.
"""

import os
from dataclasses import dataclass
from functools import lru_cache


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # ---- App ----
    app_name: str = os.getenv("APP_NAME", "SU7 CarVoice Fusion")
    app_env: str = os.getenv("APP_ENV", "dev")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # ---- Orchestrator thresholds ----
    task_confidence_threshold: float = float(os.getenv("TASK_CONFIDENCE_THRESHOLD", "0.75"))
    faq_confidence_threshold: float = float(os.getenv("FAQ_CONFIDENCE_THRESHOLD", "0.65"))
    chitchat_confidence_threshold: float = float(os.getenv("CHITCHAT_CONFIDENCE_THRESHOLD", "0.60"))

    # ---- LLM ----
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock")  # mock | doubao | vllm | openai
    doubao_api_key: str = os.getenv("DOUBAO_API_KEY", "")
    doubao_endpoint: str = os.getenv("DOUBAO_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3")
    doubao_model: str = os.getenv("DOUBAO_MODEL", "ep-20240601170316-5dhwt")
    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")

    # ---- NLP service URLs ----
    nlu_url: str = os.getenv("NLU_URL", "")
    reject_url: str = os.getenv("REJECT_URL", "")

    # ---- Redis ----
    redis_url: str = os.getenv("REDIS_URL", "")

    # ---- Knowledge / RAG ----
    knowledge_top_k: int = int(os.getenv("KNOWLEDGE_TOP_K", "3"))
    web_search_enabled: bool = _to_bool(os.getenv("WEB_SEARCH_ENABLED"), False)
    knowledge_docs_path: str = os.getenv("KNOWLEDGE_DOCS_PATH", "data/knowledge/su7_docs.json")
    retriever_backend: str = os.getenv("RETRIEVER_BACKEND", "mock")  # mock | bm25 | faiss | milvus
    reranker_backend: str = os.getenv("RERANKER_BACKEND", "mock")   # mock | minicpm
    milvus_host: str = os.getenv("MILVUS_HOST", "127.0.0.1")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))

    # ---- MCP ----
    amap_api_key: str = os.getenv("AMAP_API_KEY", "")

    # ---- Data ----
    abbr_csv_path: str = os.getenv("ABBR_CSV_PATH", "data/abbr/abbr_ch.csv")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
