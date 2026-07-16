from dataclasses import dataclass
from typing import Callable

from app.services.skills.handlers import (
    charging,
    climate,
    media_control,
    navigation,
    safety,
    vehicle_status,
    window,
)


@dataclass(frozen=True)
class SkillSpec:
    name: str
    risk_level: str
    category: str
    description: str
    keywords: tuple[str, ...]
    handler: Callable[[str], str]


class SkillsRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillSpec] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        for spec in (
            SkillSpec(
                "media_control",
                "low",
                "infotainment",
                "控制媒体播放与切歌",
                ("播放", "暂停", "下一首", "音乐"),
                media_control.handle,
            ),
            SkillSpec(
                "navigate_to",
                "medium",
                "navigation",
                "执行导航与目的地跳转",
                ("导航", "前往", "去"),
                navigation.handle,
            ),
            SkillSpec(
                "vehicle_status",
                "low",
                "vehicle",
                "查询车辆健康、电量与状态",
                ("电量", "胎压", "续航", "状态"),
                vehicle_status.handle,
            ),
            SkillSpec(
                "ac_control",
                "medium",
                "climate",
                "控制空调温度与风量",
                ("空调", "温度", "制冷", "制热"),
                climate.handle,
            ),
            SkillSpec(
                "window_control",
                "medium",
                "vehicle",
                "控制车窗开闭",
                ("车窗", "开窗", "关窗"),
                window.handle,
            ),
            SkillSpec(
                "charge_management",
                "medium",
                "energy",
                "查询充电状态与预约充电",
                ("充电", "补能", "预约充电"),
                charging.handle,
            ),
            SkillSpec(
                "sensitive_vehicle_control",
                "high",
                "safety",
                "高风险车辆控制能力（需要二次确认）",
                ("自动驾驶", "关闭安全", "解锁车辆", "远程控制"),
                safety.handle,
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

    def describe_skills(self) -> list[dict[str, str | list[str]]]:
        skills = []
        for spec in self._skills.values():
            skills.append(
                {
                    "name": spec.name,
                    "risk_level": spec.risk_level,
                    "category": spec.category,
                    "description": spec.description,
                    "keywords": list(spec.keywords),
                }
            )
        return skills
