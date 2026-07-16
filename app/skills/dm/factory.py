"""Dialog Manager factory — routes skill names to handler functions.

Ported from CarVoice_Agent/function_call/dm/factory.py.
"""

from collections.abc import Callable
from typing import Any


class DMFactory:
    """Registry mapping domain names to their async process functions."""

    _handlers: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator: register a handler under *name*."""
        def decorator(func: Callable) -> Callable:
            cls._handlers[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str) -> Callable | None:
        return cls._handlers.get(name)

    @classmethod
    def list_domains(cls) -> list[str]:
        return list(cls._handlers.keys())


# Register built-in handlers
def _register_defaults() -> None:
    from app.skills.dm import maps, music, weather  # noqa: F401

_register_defaults()
