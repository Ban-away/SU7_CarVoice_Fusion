"""CarVoice 推理服务 — Intent (8008), Reject (8007), NLU (8009)。

Ported from CarVoice_Agent/train/intent_infer.py + reject_infer.py +
function_call/chatnlu_infer.py。

使用 FastAPI 替代原始 Flask。可直接作为独立服务运行，也可被 NUL_URL/REJECT_URL 配置对接。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
import torch

from app.train.models import BertClassifier, BertTinyClassifier

logger = logging.getLogger(__name__)

# ── Shared helpers ────────────────────────────────────────────────


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_class_list() -> list[str]:
    path = Path("data/nlu/class_labels.txt")
    if path.exists():
        return [l.strip().split(":")[-1] if ":" in l else l.strip()
                for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return []


def _load_intent_map() -> dict[str, str]:
    path = Path("data/nlu/intent_map.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _load_slot_map() -> dict:
    path = Path("data/nlu/slot_intent.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}

def _load_tools() -> list[dict]:
    try:
        from app.skills.definitions import TOOLS
        return TOOLS
    except ImportError:
        return []


# ── Intent Server (port 8008) ───────────────────────────────────

intent_app = FastAPI(title="Intent Classifier", version="1.0")
_intent_device = _get_device()
_intent_model: BertClassifier | None = None
_intent_class_list: list[str] = []


class IntentRequest(BaseModel):
    query: str = Field(..., min_length=1)
    trace_id: str = Field(default="")


class IntentResponse(BaseModel):
    intent_ids: list[int] = Field(default_factory=list)
    intents: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)


@intent_app.on_event("startup")
def _load_intent_model():
    global _intent_model, _intent_class_list
    ckpt_path = os.getenv("INTENT_MODEL_PATH", "models/saved/intent/bert.ckpt")
    _intent_class_list = _load_class_list()
    model_path = os.getenv("INTENT_BERT_PATH", "models/chinese_roberta_wwm_ext")
    if not os.path.exists(ckpt_path):
        logger.warning("Intent model checkpoint not found: %s — using untrained model", ckpt_path)

    _intent_model = BertClassifier(len(_intent_class_list), model_path=model_path).to(_intent_device)
    if os.path.exists(ckpt_path):
        _intent_model.load_state_dict(torch.load(ckpt_path, map_location=_intent_device))
    _intent_model.eval()
    logger.info("Intent service ready (classes=%d)", len(_intent_class_list))


@intent_app.post("/intent-server/v1", response_model=IntentResponse)
def predict_intent(req: IntentRequest):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("models/chinese_roberta_wwm_ext")
    enc = tokenizer(req.query, max_length=32, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = _intent_model(enc["input_ids"].to(_intent_device), enc["attention_mask"].to(_intent_device))
    probs = torch.softmax(logits, dim=-1)[0]
    top5 = torch.topk(probs, min(5, len(probs)))
    ids = top5.indices.cpu().tolist()
    return IntentResponse(
        intent_ids=ids,
        intents=[_intent_class_list[i] if i < len(_intent_class_list) else str(i) for i in ids],
        scores=[round(float(s), 4) for s in top5.values.cpu().tolist()],
    )


# ── Reject Server (port 8007) ───────────────────────────────────

reject_app = FastAPI(title="Reject Classifier", version="1.0")
_reject_device = _get_device()
_reject_model: BertTinyClassifier | None = None


class RejectRequest(BaseModel):
    query: str = Field(..., min_length=1)
    thres: float = Field(default=0.5)
    trace_id: str = Field(default="")


class RejectResponse(BaseModel):
    data: int = Field(default=1)  # 1=accept, 0=reject


@reject_app.on_event("startup")
def _load_reject_model():
    global _reject_model
    ckpt_path = os.getenv("REJECT_MODEL_PATH", "models/saved/reject/bert_tiny.ckpt")
    model_path = os.getenv("REJECT_BERT_PATH", "models/roberta_tiny_clue")
    _reject_model = BertTinyClassifier(model_path=model_path).to(_reject_device)
    if os.path.exists(ckpt_path):
        _reject_model.load_state_dict(torch.load(ckpt_path, map_location=_reject_device))
    _reject_model.eval()
    logger.info("Reject service ready")


@reject_app.post("/reject-server/v1", response_model=RejectResponse)
def predict_reject(req: RejectRequest):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("models/roberta_tiny_clue")
    enc = tokenizer(req.query, max_length=32, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = _reject_model(enc["input_ids"].to(_reject_device), enc["attention_mask"].to(_reject_device))
    probs = torch.softmax(logits, dim=-1)[0]
    return RejectResponse(data=1 if probs[1].item() > req.thres else 0)


# ── NLU Server (port 8009) ──────────────────────────────────────

nlu_app = FastAPI(title="NLU Function Calling", version="1.0")


class NLURequest(BaseModel):
    query: str = Field(..., min_length=1)
    trace_id: str = Field(default="")
    enable_dm: bool = Field(default=True)


@nlu_app.post("/chatnlu-server/v1")
def predict_nlu(req: NLURequest):
    from app.skills.slot_processor import process_nlu_result
    from app.llm.base import LLMMessage, create_llm_client
    from app.prompts.nlu import NLU_SYSTEM_PROMPT

    class_list = _load_class_list()
    intent_map = _load_intent_map()
    slot_map = _load_slot_map()
    tools = _load_tools()

    # 1. Intent recall via BERT
    top5_ids = []
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("models/chinese_roberta_wwm_ext")
        enc = tokenizer(req.query, max_length=32, padding="max_length", truncation=True, return_tensors="pt")
        if _intent_model is not None:
            with torch.no_grad():
                logits = _intent_model(enc["input_ids"].to(_intent_device), enc["attention_mask"].to(_intent_device))
            probs = torch.softmax(logits, dim=-1)[0]
            top5_ids = torch.topk(probs, min(5, len(probs))).indices.cpu().tolist()
    except Exception:
        pass

    # 2. LLM function calling
    try:
        settings = __import__("app.shared.config", fromlist=["get_settings"]).get_settings()
        llm = create_llm_client(settings.llm_provider)
        messages = [
            LLMMessage(role="system", content=NLU_SYSTEM_PROMPT),
            LLMMessage(role="user", content=req.query),
        ]
        resp = llm.chat(messages, temperature=0.0, max_tokens=256)

        function = [{
            "function": {
                "name": "Unknown",
                "arguments": "{}",
            }
        }]
        try:
            parsed = json.loads(resp.content) if resp.content.strip().startswith("[") else {"name": "Unknown", "arguments": {}}
        except json.JSONDecodeError:
            parsed = {"name": "Unknown", "arguments": {}}

        result_str = process_nlu_result(
            function,
            intent_map if intent_map else {},
            slot_map if slot_map else {},
        )
        return {"function": parsed.get("name", "Unknown"), "slots": parsed.get("arguments", {}), "intent": result_str}

    except Exception:
        logger.exception("NLU failed")
        top1_id = top5_ids[0] if top5_ids else 0
        func_name = class_list[top1_id] if top1_id < len(class_list) else "Unknown"
        return {"function": func_name, "slots": {}, "intent": func_name}
