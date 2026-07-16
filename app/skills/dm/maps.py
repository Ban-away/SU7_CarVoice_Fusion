"""Maps / Navigation skill handler.

Ported from CarVoice_Agent/function_call/dm/maps.py.
"""

import logging

from app.skills.dm.factory import DMFactory

logger = logging.getLogger(__name__)


@DMFactory.register("maps")
async def process(func_name: str, query: str, slots: dict) -> tuple[dict, str]:
    """Handle navigation-related intents (Go_POI, etc.).

    Returns (raw_result, nlg_text).
    """
    keyword = slots.get("keyword", slots.get("poi", ""))
    city = slots.get("city", "")

    # In production this would call the MCP Amap server
    if keyword:
        result_text = f"已搜索到「{keyword}」的相关地点"
    else:
        result_text = "已执行导航操作"

    return {"status": "ok", "keyword": keyword, "city": city}, result_text
