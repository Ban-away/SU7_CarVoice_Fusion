import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.orchestrator.router import ChatOrchestrator

router = APIRouter()
orchestrator = ChatOrchestrator()


@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            incoming = await websocket.receive_text()
            confirm = False
            message = incoming

            try:
                payload = json.loads(incoming)
                if isinstance(payload, dict):
                    message = str(payload.get("message", "")).strip()
                    confirm = bool(payload.get("confirm", False))
            except json.JSONDecodeError:
                pass

            if not message:
                await websocket.send_json(
                    {
                        "type": "error",
                        "text": "message 不能为空",
                        "citations": [],
                        "trace": {
                            "route": "Unknown",
                            "latency_ms": 0,
                            "fallback_reason": "empty_message",
                        },
                    }
                )
                continue

            response = orchestrator.handle(message, confirm=confirm)
            await websocket.send_json(response.model_dump())
    except WebSocketDisconnect:
        return
