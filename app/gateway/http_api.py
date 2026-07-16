from fastapi import APIRouter

from app.orchestrator.router import ChatOrchestrator
from app.shared.schemas import (
    ChatRequest,
    ChatResponse,
    KnowledgeRetrieveRequest,
    KnowledgeRetrieveResponse,
    SkillInfo,
)

router = APIRouter()
orchestrator = ChatOrchestrator()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return orchestrator.handle(request.message, confirm=request.confirm, session_id=request.session_id)


@router.get("/api/v1/skills", response_model=list[SkillInfo])
def list_skills() -> list[SkillInfo]:
    return [SkillInfo(**item) for item in orchestrator.skills_registry.describe_skills()]


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
