from dataclasses import dataclass
from typing import Callable

from app.services.skills import handlers


@dataclass(frozen=True)
class SkillSpec:
    name: str
    risk_level: str
    keywords: tuple[str, ...]
    handler: Callable[[str], str]


class SkillsRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillSpec] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        for spec in (
            SkillSpec("media_control", "low", ("播放", "暂停", "下一首", "音乐"), handlers.media_control),
            SkillSpec("navigate_to", "medium", ("导航", "前往", "去"), handlers.navigate_to),
            SkillSpec("vehicle_status", "low", ("电量", "胎压", "续航", "状态"), handlers.vehicle_status),
            SkillSpec(
                "sensitive_vehicle_control",
                "high",
                ("自动驾驶", "关闭安全", "解锁车辆", "远程控制"),
                handlers.sensitive_vehicle_control,
            ),
        ):
            self._skills[spec.name] = spec

    def resolve_skill(self, message: str) -> SkillSpec | None:
        for spec in self._skills.values():
            if any(keyword in message for keyword in spec.keywords):
                return spec
        return None

    def execute(self, skill_name: str, message: str) -> str:
        if skill_name not in self._skills:
            raise ValueError(f"Skill '{skill_name}' is not registered in whitelist")
        return self._skills[skill_name].handler(message)

    def list_skills(self) -> list[str]:
        return list(self._skills.keys())
