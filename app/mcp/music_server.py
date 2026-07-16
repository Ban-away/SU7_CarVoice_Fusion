"""QQ Music MCP server.

Exposes a ``search_music`` tool that queries QQ Music by keyword and returns
track metadata (id, mid, name, subtitle, etc.).
"""

import asyncio
from typing import Any, Dict, List

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "The 'mcp' package is required for the QQ Music MCP server. "
        "Install it with: pip install mcp fastmcp"
    )
from qqmusic_api import search

from app.shared.logging import get_logger

logger = get_logger(__name__)

mcp = FastMCP("mcp-qqmusic-server")


@mcp.tool()
async def search_music(
    keyword: str, page: int = 1, num: int = 3
) -> List[Dict[str, Any]]:
    """Search QQ Music for tracks matching the given keyword.

    Args:
        keyword: Search keyword or phrase.
        page: Page number for pagination (default 1).
        num: Maximum number of results to return (default 3).

    Returns:
        A list of track dicts with keys: id, mid, name, pmid, icon_url,
        subtitle, time_public, title.
    """
    logger.info("Searching QQ Music: keyword=%s, page=%s, num=%s", keyword, page, num)

    result = await search.search_by_type(keyword=keyword, page=page, num=num)

    if not isinstance(result, list):
        return []

    filtered: List[Dict[str, Any]] = []
    for item in result:
        song_info: Dict[str, Any] = {
            "id": item.get("id"),
            "mid": item.get("mid"),
            "name": item.get("name"),
            "pmid": item.get("pmid", ""),
            "icon_url": item.get("icon_url", ""),
            "subtitle": item.get("subtitle", ""),
            "time_public": item.get("time_public", ""),
            "title": item.get("title", item.get("name", "")),
        }
        filtered.append(song_info)

    return filtered


if __name__ == "__main__":
    mcp.run(transport="stdio")
