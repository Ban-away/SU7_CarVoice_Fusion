from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


ResponseType = Literal["task_result", "faq_answer", "chitchat", "clarification", "error"]
RouteType = Literal["Task", "FAQ", "Chitchat", "Unknown"]


class Citation(BaseModel):
    source: str
    page: Optional[int] = None


class Trace(BaseModel):
    route: RouteType
    classifier_confidence: Optional[float] = None
    knowledge_hit_count: Optional[int] = None
    latency_ms: int = 0
    fallback_reason: Optional[str] = None
    risk_level: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    confirm: bool = False

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message 不能为空")
        return cleaned


class ChatResponse(BaseModel):
    type: ResponseType
    text: str
    citations: list[Citation] = Field(default_factory=list)
    trace: Trace
