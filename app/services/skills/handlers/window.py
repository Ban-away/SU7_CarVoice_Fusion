"""车窗控制"""


def handle(message: str) -> str:
    """控制车窗开闭。"""
    if "打开" in message:
        return "已打开车窗。"
    if "关闭" in message:
        return "已关闭车窗。"
    return "已执行车窗控制。"
