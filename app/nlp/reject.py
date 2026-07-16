"""Rejection model — determines if a query should be rejected (out-of-domain).

Ported from CarVoice_Agent/client/reject.py.
"""

import logging

import httpx

from app.shared.config import get_settings

logger = logging.getLogger(__name__)


def should_reject(query: str, trace_id: str = "") -> bool:
    """Check whether *query* should be rejected.

    Calls the external reject service when configured; defaults to
    accepting everything (not rejected) when the service is unavailable.
    """
    settings = get_settings()
    if not settings.reject_url:
        # No reject service → accept everything
        return False

    try:
        resp = httpx.post(
            settings.reject_url,
            json={"query": query, "trace_id": trace_id},
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()
        # Threshold 0.5: >0.5 means reject
        score = float(data.get("score", 0.0))
        return score > 0.5
    except Exception:
        logger.exception("Reject service call failed — defaulting to accept")
        return False
