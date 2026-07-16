"""Weather query skill handler.

Ported from CarVoice_Agent/function_call/dm/weather.py.
"""

import logging

from app.skills.dm.factory import DMFactory

logger = logging.getLogger(__name__)


@DMFactory.register("weather")
async def process(func_name: str, query: str, slots: dict) -> tuple[dict, str]:
    """Handle weather-related intents (Query_Weather, etc.).

    Returns (raw_result, nlg_text).
    """
    city = slots.get("city", "北京")
    date = slots.get("date", "今天")

    result_text = f"{city}{date}天气：晴，18~25℃，空气质量良好"

    return {"status": "ok", "city": city, "date": date}, result_text
