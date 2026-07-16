"""媒体播放控制"""


def handle(message: str) -> str:
    """控制媒体播放与切歌。"""
    if "暂停" in message:
        return "已暂停当前媒体播放。"
    if "下一首" in message:
        return "已切换到下一首。"
    return "已执行媒体控制指令。"
