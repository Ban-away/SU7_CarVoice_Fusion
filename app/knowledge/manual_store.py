"""MongoDB 用户手册存储 — Pydantic 模型 + 连接管理。

Ported from XIAOMI_SU7_RAG/src/fields/manual_info_mongo.py + manual_images.py。
可选依赖 — MongoDB 不可用时优雅降级。
"""

from __future__ import annotations

import logging
import os
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Pydantic 模型 ────────────────────────────────────────────────


class ManualImage(BaseModel):
    title: str = ""
    url: str = ""
    page: int | None = None
    width: int = 0
    height: int = 0


class ManualText(BaseModel):
    unique_id: str
    page_content: str
    metadata: dict = Field(default_factory=dict)
    page: int | None = None
    source: str = ""


# ── MongoDB 连接 ──────────────────────────────────────────────────

_mongo_client: Any = None
_mongo_db: Any = None


def _get_config():
    return {
        "host": os.getenv("MONGO_HOST", "localhost"),
        "port": int(os.getenv("MONGO_PORT", "27017")),
        "db": os.getenv("MONGO_DB_NAME", "mydatabase"),
        "user": os.getenv("MONGO_USERNAME", ""),
        "pass": os.getenv("MONGO_PASSWORD", ""),
        "auth_source": os.getenv("MONGO_AUTH_SOURCE", "admin"),
    }


def get_collection(name: str = "manual_text"):
    """Get or create MongoDB collection. Returns None if unavailable."""
    global _mongo_client, _mongo_db
    if _mongo_db is not None:
        return _mongo_db[name]

    try:
        from pymongo import MongoClient
        cfg = _get_config()
        uri = f"mongodb://{cfg['host']}:{cfg['port']}"
        if cfg["user"]:
            uri = f"mongodb://{cfg['user']}:{cfg['pass']}@{cfg['host']}:{cfg['port']}/?authSource={cfg['auth_source']}"
        _mongo_client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        _mongo_client.server_info()  # Test connection
        _mongo_db = _mongo_client[cfg["db"]]
        logger.info("MongoDB connected: %s/%s", cfg["host"], cfg["db"])
        return _mongo_db[name]
    except Exception:
        logger.warning("MongoDB unavailable — manual storage disabled")
        return None


def save_manual_page(doc: ManualText, collection_name: str = "manual_text") -> bool:
    """Save a manual page to MongoDB."""
    col = get_collection(collection_name)
    if col is None:
        return False
    try:
        col.update_one({"unique_id": doc.unique_id}, {"$set": doc.model_dump()}, upsert=True)
        return True
    except Exception:
        logger.exception("Failed to save manual page")
        return False


def find_by_id(unique_id: str, collection_name: str = "manual_text") -> dict | None:
    """Look up a manual page by unique_id."""
    col = get_collection(collection_name)
    if col is None:
        return None
    return col.find_one({"unique_id": unique_id})
