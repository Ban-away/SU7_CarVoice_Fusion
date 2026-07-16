"""Abbreviation expansion using a domain-specific term mapping.

Ported from XIAOMI_SU7_RAG/src/gen_qa/qa_filter.py (abbreviation logic).
"""

import csv
import logging
from pathlib import Path

from app.shared.config import get_settings

logger = logging.getLogger(__name__)

# Built-in automotive abbreviation table (fallback)
_BUILTIN_ABBR: dict[str, str] = {
    "ECU": "电子控制单元",
    "PEPS": "无钥匙进入和无钥匙启动",
    "TPMS": "轮胎压力监测系统",
    "EPB": "电子驻车制动系统",
    "V2X": "车联网通信",
    "ABS": "防抱死制动系统",
    "ESP": "电子稳定程序",
    "ACC": "自适应巡航控制",
    "LKA": "车道保持辅助",
    "AEB": "自动紧急制动",
    "HUD": "抬头显示",
    "DMS": "驾驶员监测系统",
    "OTA": "空中升级",
    "SOC": "电池电量状态",
    "BMS": "电池管理系统",
    "VCU": "整车控制器",
    "MCU": "微控制器单元",
    "IVI": "车载信息娱乐系统",
    "TBOX": "车载通信终端",
    "ADAS": "高级驾驶辅助系统",
}


def load_abbr_map(csv_path: str | None = None) -> dict[str, str]:
    """Load abbreviation→full-name map from CSV, falling back to built-in table."""
    path_str = csv_path or get_settings().abbr_csv_path
    path = Path(path_str)
    if not path.exists():
        logger.info("Abbreviation CSV not found at %s — using built-in table", path_str)
        return dict(_BUILTIN_ABBR)

    abbr_map = dict(_BUILTIN_ABBR)
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    abbr = row[0].strip()
                    full = row[1].strip()
                    if abbr and full:
                        abbr_map[abbr] = full
    except Exception:
        logger.exception("Failed to load abbreviation CSV")

    return abbr_map


def expand_abbreviations(text: str, abbr_map: dict[str, str] | None = None) -> str:
    """Replace known abbreviations in *text* with their Chinese full names."""
    if abbr_map is None:
        abbr_map = load_abbr_map()

    result = text
    for abbr, full in sorted(abbr_map.items(), key=lambda x: -len(x[0])):
        if abbr in result:
            result = result.replace(abbr, full)
    return result
