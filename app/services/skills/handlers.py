def media_control(message: str) -> str:
    if "暂停" in message:
        return "已暂停当前媒体播放。"
    if "下一首" in message:
        return "已切换到下一首。"
    return "已执行媒体控制指令。"


def navigate_to(message: str) -> str:
    destination = "目的地"
    for split_token in ["到", "去", "前往"]:
        if split_token in message:
            candidate = message.split(split_token, 1)[-1].strip(" 。，,.！!")
            if candidate:
                destination = candidate
                break
    return f"已开始导航到{destination}。"


def vehicle_status(message: str) -> str:
    if "电量" in message or "续航" in message:
        return "当前电量约 78%，预计续航约 520km。"
    return "车辆状态正常，胎压与车门状态均正常。"


def ac_control(message: str) -> str:
    if "调高" in message or "升高" in message:
        return "已将空调温度调高 1℃。"
    if "调低" in message or "降低" in message:
        return "已将空调温度调低 1℃。"
    return "已执行空调控制。"


def window_control(message: str) -> str:
    if "打开" in message:
        return "已打开车窗。"
    if "关闭" in message:
        return "已关闭车窗。"
    return "已执行车窗控制。"


def charge_management(message: str) -> str:
    if "预约" in message:
        return "已为你创建充电预约。"
    return "已查询充电管理信息。"


def sensitive_vehicle_control(message: str) -> str:
    return "高风险车辆控制指令已接收，待二次确认后执行。"
