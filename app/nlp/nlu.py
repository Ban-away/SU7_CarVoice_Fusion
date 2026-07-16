"""NLU — intent + slot extraction via external NLU service.

Ported from CarVoice_Agent/client/nlu.py.
"""

import json
import logging

import httpx

from app.shared.config import get_settings

logger = logging.getLogger(__name__)


def extract_intent(query: str, trace_id: str = "", enable_dm: bool = True) -> dict:
    """Call the external NLU service to extract intent, function, and slots.

    Returns a dict with keys: function, slots, intent, etc.
    Falls back to a safe default on failure.
    """
    settings = get_settings()
    if not settings.nlu_url:
        return _fallback_nlu(query)

    try:
        resp = httpx.post(
            settings.nlu_url,
            json={
                "query": query,
                "trace_id": trace_id,
                "enable_dm": enable_dm,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.exception("NLU service call failed")
        return _fallback_nlu(query)


def _fallback_nlu(query: str) -> dict:
    """Simple keyword-based NLU fallback."""
    # Basic keyword matching
    func_map = {
        "导航": {"function": "Go_POI", "intent": "NAVIGATION"},
        "播放": {"function": "Search_Music", "intent": "MUSIC"},
        "天气": {"function": "Query_Weather", "intent": "WEATHER"},
        "空调": {"function": "Open_Air_Condition", "intent": "VEHICLE_CONTROL"},
        "车窗": {"function": "Open_Window", "intent": "VEHICLE_CONTROL"},
        "电话": {"function": "Call_Phone", "intent": "PHONE"},
    }
    for kw, func in func_map.items():
        if kw in query:
            return {
                "function": func["function"],
                "slots": {},
                "intent": func["intent"],
            }
    return {"function": "Unknown", "slots": {}, "intent": "UNKNOWN"}
