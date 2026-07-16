"""空调与温度控制"""


def handle(message: str) -> str:
    """控制空调温度与风量。"""
    if "调高" in message or "升高" in message:
        return "已将空调温度调高 1℃。"
    if "调低" in message or "降低" in message:
        return "已将空调温度调低 1℃。"
    return "已执行空调控制。"
