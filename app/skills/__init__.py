"""Skills service — task execution, 455 function definitions, DM handlers, slot processing.

- ``registry.py``: Skill whitelist with risk levels and keyword matching.
- ``definitions.py``: 455 canonical function/tool schemas (ported from CarVoice_Agent).
- ``slot_processor.py``: Slot value normalisation (position mapping, extreme extraction, etc.).
- ``dm/``: Dialog Manager handlers for maps, music, and weather.
- ``handlers/``: Individual skill handler implementations.
"""

from app.services.skills.registry import SkillSpec, SkillsRegistry  # noqa: F401
