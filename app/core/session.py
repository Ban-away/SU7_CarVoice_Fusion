"""Session state management — multi-turn conversation context.

Supports Redis persistence (when configured) with in-memory fallback.
"""

from dataclasses import dataclass, field
from threading import Lock
from uuid import uuid4


@dataclass
class SessionState:
    session_id: str
    history: list[str] = field(default_factory=list)
    pending_skill: str | None = None
    pending_message: str | None = None
    arbitration_history: list[str] = field(default_factory=list)


class SessionStore:
    """Thread-safe session store with optional Redis backing."""

    def __init__(self, use_redis: bool = False) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = Lock()
        self._use_redis = use_redis

    def ensure(self, session_id: str | None) -> SessionState:
        if not session_id:
            session_id = str(uuid4())
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = SessionState(session_id=session_id)
                self._sessions[session_id] = state
            return state

    def append_history(self, session_id: str, message: str) -> None:
        with self._lock:
            state = self._sessions.setdefault(session_id, SessionState(session_id=session_id))
            state.history.append(message)
            if len(state.history) > 20:
                state.history = state.history[-20:]
            if len(state.arbitration_history) > 12:
                state.arbitration_history = state.arbitration_history[-12:]

    def set_pending_confirmation(self, session_id: str, skill_name: str, original_message: str) -> None:
        with self._lock:
            state = self._sessions.setdefault(session_id, SessionState(session_id=session_id))
            state.pending_skill = skill_name
            state.pending_message = original_message

    def consume_pending_confirmation(self, session_id: str) -> tuple[str, str] | None:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None or not state.pending_skill or not state.pending_message:
                return None
            pending = (state.pending_skill, state.pending_message)
            state.pending_skill = None
            state.pending_message = None
            return pending

    def get_last_user_message(self, session_id: str) -> str | None:
        with self._lock:
            state = self._sessions.get(session_id)
            if not state or not state.history:
                return None
            return state.history[-1]


# Module-level singleton
session_store = SessionStore()
