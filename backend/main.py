"""
FastAPI entrypoint.

  uvicorn backend.main:app --reload --port 8000

Wires every router under /api, applies open CORS for the Vite dev server,
initialises SQLite + the dense backend probe at startup, and provides a
single uniform error envelope.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.db import init_db
from backend.routes.agent import router as agent_router
from backend.routes.documents import router as documents_router
from backend.routes.eval import router as eval_router
from backend.routes.rag import router as rag_router
from backend.routes.sessions import router as sessions_router
from backend.routes.settings import router as settings_router
from backend.vector_store import get_backend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("legal-rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        get_backend()  # probe Qdrant + hydrate from SQLite
    except Exception as e:
        # Hydration failures are non-fatal — the user might just need to
        # set their OpenAI key in Settings before the embeddings work.
        log.warning("vector backend probe failed: %s", e)
    log.info("✅ LexAI / Legal RAG API ready")
    yield


app = FastAPI(
    title="LexAI — Legal RAG API",
    version="1.0.0",
    description="Production-grade legal RAG, evaluation, and agentic workflow.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Uniform error envelope ──────────────────────────────────────
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail if isinstance(exc.detail, str) else "Request failed"},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": "Validation failed", "details": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"error": str(exc)})


# ── Routers ─────────────────────────────────────────────────────
app.include_router(settings_router)
app.include_router(documents_router)
app.include_router(rag_router)
app.include_router(sessions_router)
app.include_router(eval_router)
app.include_router(agent_router)


@app.get("/")
def root():
    return {"name": "LexAI", "status": "ok", "docs": "/docs"}
