"""Music search skill handler.

Ported from CarVoice_Agent/function_call/dm/music.py.
"""

import logging

from app.skills.dm.factory import DMFactory

logger = logging.getLogger(__name__)


@DMFactory.register("music")
async def process(func_name: str, query: str, slots: dict) -> tuple[dict, str]:
    """Handle music-related intents (Search_Music, etc.).

    Returns (raw_result, nlg_text).
    """
    keyword = slots.get("keyword", slots.get("song", ""))
    artist = slots.get("artist", "")
    genre = slots.get("genre", "流行")

    result_text = f"已搜索到音乐「{keyword or artist or genre}」"

    return {"status": "ok", "keyword": keyword, "artist": artist}, result_text
