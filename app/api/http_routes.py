"""HTTP REST API routes — chat, health, skills, knowledge, admin."""

from fastapi import APIRouter

from app.core.orchestrator import ChatOrchestrator
from app.shared.schemas import (
    ChatRequest,
    ChatResponse,
    KnowledgeRetrieveRequest,
    KnowledgeRetrieveResponse,
    SkillInfo,
)

router = APIRouter()
orchestrator = ChatOrchestrator()


# ── Health ────────────────────────────────────────────────────────────


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


# ── Chat ──────────────────────────────────────────────────────────────


@router.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return orchestrator.handle(
        request.message,
        confirm=request.confirm,
        session_id=request.session_id,
    )


# ── Skills ────────────────────────────────────────────────────────────


@router.get("/api/v1/skills", response_model=list[SkillInfo])
def list_skills() -> list[SkillInfo]:
    return [SkillInfo(**item) for item in orchestrator.skills_registry.describe_skills()]


# ── Knowledge ─────────────────────────────────────────────────────────


@router.post("/api/v1/knowledge/retrieve", response_model=KnowledgeRetrieveResponse)
def retrieve_knowledge(request: KnowledgeRetrieveRequest) -> KnowledgeRetrieveResponse:
    docs = orchestrator.knowledge_service.retrieve(request.query, top_k=request.top_k)
    answer, citations = orchestrator.knowledge_service.synthesize_with_citations(request.query, docs)
    return KnowledgeRetrieveResponse(
        query=request.query,
        answer=answer,
        citations=citations,
        hit_count=len(docs),
    )


# ── Admin: function definitions ───────────────────────────────────────


@router.get("/api/v1/functions")
def list_functions() -> dict:
    """Return the full 450+ function/tool definitions (from CarVoice_Agent)."""
    try:
        from app.skills.definitions import TOOLS
        return {"count": len(TOOLS), "tools": TOOLS}
    except ImportError:
        return {"count": 0, "tools": []}
