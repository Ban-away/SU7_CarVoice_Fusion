"""CarVoice BERT 意图识别 — 439 类函数级分类，与 CarVoice_Agent 一致。"""

from __future__ import annotations

import logging
import os
import json
from pathlib import Path

logger = logging.getLogger(__name__)

_INTENT_MODEL = None
_INTENT_CLASS_LIST: list[str] = []
_INTENT_MAP: dict[str, str] = {}


def _load_intent_model():
    """懒加载：首次调用时加载 BERT 意图模型。"""
    global _INTENT_MODEL, _INTENT_CLASS_LIST, _INTENT_MAP
    if _INTENT_MODEL is not None:
        return True

    ckpt_path = os.getenv("INTENT_MODEL_PATH", "models/saved/intent/bert.ckpt")
    if not Path(ckpt_path).exists():
        return False

    try:
        import torch
        from app.train.core import BertModel, BertTokenizer
        from app.train.models import Model
        import torch.nn as nn

        # 加载分类标签
        class_path = Path("data/nlu/class_labels.txt")
        if class_path.exists():
            with open(class_path, encoding="utf-8") as f:
                _INTENT_CLASS_LIST = [l.strip().split(":")[-1] if ":" in l else l.strip()
                                      for l in f if l.strip()]

        # 加载意图映射
        map_path = Path("data/nlu/intent_map.json")
        if map_path.exists():
            _INTENT_MAP = json.loads(map_path.read_text(encoding="utf-8"))

        num_classes = len(_INTENT_CLASS_LIST)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = Model(num_classes, model_path="models/chinese_roberta_wwm_ext").to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        _INTENT_MODEL = (model, device)
        logger.info("BERT intent model loaded: %d classes, device=%s", num_classes, device)
        return True
    except Exception:
        logger.warning("BERT intent model not available — using rule/LLM fallback")
        return False


def predict_intent(text: str) -> tuple[str, str] | None:
    """用 BERT 模型预测意图函数名。返回 (function_name, intent_name) 或 None。"""
    if not _load_intent_model():
        return None

    try:
        model, device = _INTENT_MODEL
        from app.train.core import BertTokenizer

        tokenizer = BertTokenizer.from_pretrained("models/chinese_roberta_wwm_ext")
        tokens = tokenizer.tokenize(text)
        tokens = (["[CLS]"] + tokens[:30] + ["[SEP]"])[:32]
        ids = tokenizer.convert_tokens_to_ids(tokens)
        mask = [1] * len(ids) + [0] * (32 - len(ids))
        ids = ids + [0] * (32 - len(ids))

        import torch
        with torch.no_grad():
            tensor_ids = torch.tensor([ids]).to(device)
            tensor_mask = torch.tensor([mask]).to(device)
            logits = model((tensor_ids, torch.tensor([len(tokens)]), tensor_mask))

        # Top-1 预测
        idx = torch.argmax(logits, dim=1).item()
        if idx < len(_INTENT_CLASS_LIST):
            func_name = _INTENT_CLASS_LIST[idx]
            intent_name = _INTENT_MAP.get(str(idx), func_name) if _INTENT_MAP else func_name
            return func_name, intent_name
    except Exception:
        logger.exception("Intent prediction failed")
    return None
