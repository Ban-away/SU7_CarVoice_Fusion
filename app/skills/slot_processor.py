"""Slot value normalisation — position mapping, extreme extraction, percentage conversion.

Ported from CarVoice_Agent/function_call/slot_process.py.
"""

import json
import logging

logger = logging.getLogger(__name__)

# Position mapping: Chinese → English enum
POSITION_MAP: dict[str, str] = {
    "主驾": "MAIN",
    "副驾": "VICE",
    "左侧": "LEFT",
    "右侧": "RIGHT",
    "前排": "FRONT",
    "后排": "REAR",
    "左后": "LEFT_REAR",
    "右后": "RIGHT_REAR",
    "主对角": "MAIN_DIAGONAL",
    "副对角": "VICE_DIAGONAL",
    "所有": "ALL",
    "吹脚": "FOOT",
    "吹脸": "FACE",
    "吹窗": "WINDOW",
    "吹脸吹脚": "FACE_AND_FOOT",
    "吹窗吹脚": "WINDOW_AND_FOOT",
    "左前": "MAIN",
    "右前": "VICE",
    "主副驾": "FRONT",
}

# Value normalisation: LLM output variants → canonical
VALUE_NORM: dict[str, str] = {
    "所有的": "所有",
    "主驾驶": "主驾",
    "副驾驶": "副驾",
    "司机": "主驾",
    "驾驶位": "主驾",
    "左前": "主驾",
    "右前": "副驾",
    "usb音乐": "USB音乐",
}


def normalize_slot_value(key: str, value: str) -> str:
    """Normalize a single slot value based on its key type."""
    # Generic value norm
    value = VALUE_NORM.get(value, value)

    # Position → enum
    if key in ("POSITION", "position", "Position"):
        return POSITION_MAP.get(value, value)

    # Percentage → decimal
    if key in ("ratio", "Ratio"):
        if "%" in value:
            try:
                value = str(float(value.replace("%", "")) / 100)
            except ValueError:
                pass

    # Duration → seconds-only
    if key == "对话时长":
        value = value.replace("秒", "")

    # Extreme → canonical
    if key == "Extreme":
        if value in ("最大", "最高", "最强", "最亮", "最热"):
            return "最大"
        if value in ("最小", "最低", "最弱", "最暗", "最冷"):
            return "最小"

    return value


def process_nlu_result(
    function: list[dict],
    intent_map: dict[str, str],
    slot_map: dict[str, dict[str, str]],
) -> str:
    """Convert raw NLU output into a formatted intent+slots string.

    Returns:
        String like "Go_POI-keyword:公司,city:北京" or "Unknown-无"
    """
    try:
        func_name = function[0].get("function", {}).get("name", "NULL")
        mapped_intent = intent_map.get(func_name, func_name)

        slots_raw = function[0].get("function", {}).get("arguments", "{}")
        slots = json.loads(slots_raw) if isinstance(slots_raw, str) else slots_raw

        result_parts: list[str] = [mapped_intent]
        key_remap = slot_map.get(func_name, {})

        for key, value in slots.items():
            if not value or value in ("不限", "无", "空", "none", "None", "Unknown", "unknown"):
                continue
            remapped_key = key_remap.get(key, key) if isinstance(key_remap, dict) else key
            normalized = normalize_slot_value(remapped_key, str(value))
            result_parts.append(f"{remapped_key}:{normalized}")

        if len(result_parts) == 1:
            result_parts.append("无")
        return ",".join(result_parts) if len(result_parts) > 1 else f"{mapped_intent}-无"

    except Exception:
        logger.exception("Slot processing failed")
        return "未知-无"
