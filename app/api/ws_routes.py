"""WebSocket API routes — real-time chat with streaming protocol."""

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.orchestrator import ChatOrchestrator

router = APIRouter()
orchestrator = ChatOrchestrator()


@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            incoming = await websocket.receive_text()
            confirm = False
            session_id = None
            message = incoming

            try:
                payload = json.loads(incoming)
                if isinstance(payload, dict):
                    message = str(payload.get("message", "")).strip()
                    confirm = bool(payload.get("confirm", False))
                    session_id = payload.get("session_id")
            except json.JSONDecodeError:
                pass

            if not message:
                await websocket.send_json(_error_frame("message 不能为空", "empty_message"))
                continue

            response = orchestrator.handle(message, confirm=confirm, session_id=session_id)
            await websocket.send_json(response.model_dump())

    except WebSocketDisconnect:
        return


def _error_frame(text: str, reason: str) -> dict:
    return {
        "type": "error",
        "text": text,
        "citations": [],
        "trace": {
            "route": "Unknown",
            "latency_ms": 0,
            "fallback_reason": reason,
        },
    }
