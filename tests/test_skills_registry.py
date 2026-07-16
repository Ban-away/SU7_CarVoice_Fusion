import pytest

from app.services.skills.registry import SkillsRegistry


def test_skill_registry_whitelist_only() -> None:
    registry = SkillsRegistry()

    with pytest.raises(ValueError):
        registry.execute("non_existing", "test")


def test_resolve_skill_from_keywords() -> None:
    registry = SkillsRegistry()
    skill = registry.resolve_skill("请播放音乐")

    assert skill is not None
    assert skill.name == "media_control"
