from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.gateway.http_api import router as http_router
from app.gateway.ws_api import router as ws_router
from app.shared.errors import AppError
from app.shared.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

app = FastAPI(title="SU7 CarVoice Fusion")
app.include_router(http_router)
app.include_router(ws_router)


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": "error",
            "text": exc.message,
            "citations": [],
            "trace": {
                "route": "Unknown",
                "latency_ms": 0,
                "fallback_reason": exc.code,
            },
        },
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "type": "error",
            "text": "服务内部错误，请稍后重试",
            "citations": [],
            "trace": {
                "route": "Unknown",
                "latency_ms": 0,
                "fallback_reason": "internal_error",
            },
        },
    )
