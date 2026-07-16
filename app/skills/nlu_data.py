"""NLU data loader — loads slot-intent mappings and intent map from disk.

Data files sourced from CarVoice_Agent/config/.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Default paths relative to project root
SLOT_INTENT_PATH = "data/nlu/slot_intent.json"
INTENT_MAP_PATH = "data/nlu/intent_map.json"
CLASS_LABELS_PATH = "data/nlu/class_labels.txt"


@lru_cache(maxsize=1)
def load_slot_intent_map(path: str | None = None) -> dict:
    """Load the slot-to-intent mapping (slot_intent.json).

    Maps function names to their expected slot definitions.
    Example: {"Set_Air_Condition_Temperature": {"Position": "位置", "Number": "number", ...}, ...}
    """
    path = path or SLOT_INTENT_PATH
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        logger.info("Loaded slot-intent map: %d entries from %s", len(data), path)
        return data
    except Exception:
        logger.warning("Could not load slot-intent map from %s — using empty map", path)
        return {}


@lru_cache(maxsize=1)
def load_intent_map(path: str | None = None) -> dict[str, str]:
    """Load the intent ID → function name mapping (new_map.json).

    Example: {"1": "Go_POI", "2": "Search_Music", ...}
    """
    path = path or INTENT_MAP_PATH
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        logger.info("Loaded intent map: %d entries from %s", len(data), path)
        return data
    except Exception:
        logger.warning("Could not load intent map from %s — using empty map", path)
        return {}


@lru_cache(maxsize=1)
def load_class_labels(path: str | None = None) -> dict[str, tuple[str, str]]:
    """Load class labels (class.txt).

    Format per line: <id>:<chinese_name>:<function_name>

    Returns: {id: (chinese_name, function_name), ...}
    """
    path = path or CLASS_LABELS_PATH
    result: dict[str, tuple[str, str]] = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 2)
            if len(parts) >= 3:
                result[parts[0]] = (parts[1], parts[2])
        logger.info("Loaded class labels: %d entries from %s", len(result), path)
    except Exception:
        logger.warning("Could not load class labels from %s", path)
    return result
