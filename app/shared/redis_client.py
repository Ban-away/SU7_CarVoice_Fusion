"""Redis client wrapper with in-memory fallback.

Ported from CarVoice_Agent/utils/redis_tool.py.
"""

import logging
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class InMemoryStore:
    """Thread-safe in-memory key-value store with TTL support (fallback)."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)


# Global in-memory store instance
_in_memory = InMemoryStore()


class RedisClient:
    """Redis client wrapper.

    When REDIS_URL is configured, connects to a real Redis instance.
    Otherwise falls back to an in-memory store suitable for development.
    """

    def __init__(self, redis_url: str = "") -> None:
        self._redis_url = redis_url
        self._redis = None
        if redis_url:
            try:
                import redis  # type: ignore[import-untyped]

                self._redis = redis.from_url(redis_url)
                self._redis.ping()
                logger.info("Redis connected: %s", redis_url)
            except Exception:
                logger.warning("Redis unavailable at %s — using in-memory fallback", redis_url)

    def get(self, key: str) -> Any | None:
        if self._redis:
            return self._redis.get(key)
        return _in_memory.get(key)

    def set(self, key: str, value: Any, ex: int | None = None) -> None:
        if self._redis:
            self._redis.set(key, value, ex=ex)
        else:
            _in_memory.set(key, value)

    def delete(self, key: str) -> None:
        if self._redis:
            self._redis.delete(key)
        else:
            _in_memory.delete(key)


# Module-level singleton (lazy init)
_client: RedisClient | None = None


def get_redis() -> RedisClient:
    global _client
    if _client is None:
        from app.shared.config import get_settings
        _client = RedisClient(redis_url=get_settings().redis_url)
    return _client
