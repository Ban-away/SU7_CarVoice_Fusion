from fastapi import APIRouter

from app.orchestrator.router import ChatOrchestrator
from app.shared.schemas import ChatRequest, ChatResponse

router = APIRouter()
orchestrator = ChatOrchestrator()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return orchestrator.handle(request.message, confirm=request.confirm)
