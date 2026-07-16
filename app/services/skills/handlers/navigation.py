"""导航与目的地跳转"""


def handle(message: str) -> str:
    """执行导航与目的地跳转。"""
    destination = "目的地"
    for split_token in ["到", "去", "前往"]:
        if split_token in message:
            candidate = message.split(split_token, 1)[-1].strip(" 。，,.！!")
            if candidate:
                destination = candidate
                break
    return f"已开始导航到{destination}。"
