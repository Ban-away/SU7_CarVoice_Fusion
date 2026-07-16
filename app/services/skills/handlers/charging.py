"""充电管理"""


def handle(message: str) -> str:
    """查询充电状态与预约充电。"""
    if "预约" in message:
        return "已为你创建充电预约。"
    return "已查询充电管理信息。"
