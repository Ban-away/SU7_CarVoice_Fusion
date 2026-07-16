"""技能处理函数集合。

每个子模块导出一个 ``handle(message: str) -> str`` 函数。
"""

from app.services.skills.handlers import (  # noqa: F401
    charging,
    climate,
    media_control,
    navigation,
    safety,
    vehicle_status,
    window,
)
