"""车辆状态查询"""


def handle(message: str) -> str:
    """查询车辆健康、电量与状态。"""
    if "电量" in message or "续航" in message:
        return "当前电量约 78%，预计续航约 520km。"
    return "车辆状态正常，胎压与车门状态均正常。"
