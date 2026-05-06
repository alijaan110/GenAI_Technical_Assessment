"""Settings + system health endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from backend.settings import get_settings, update_settings
from backend.vector_store import active_backend_name, get_backend, reset_backend

router = APIRouter(tags=["settings"])


class SettingsUpdate(BaseModel):
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    chunk_size: Optional[str] = None
    chunk_overlap: Optional[str] = None


@router.get("/api/settings")
def read_settings():
    s = get_settings()
    return {
        "openai_api_key": "sk-…configured" if s.openai_api_key else "",
        "anthropic_api_key": "sk-ant-…configured" if s.anthropic_api_key else "",
        "tavily_api_key": "tvly-…configured" if s.tavily_api_key else "",
        "qdrant_url": s.qdrant_url,
        "qdrant_api_key": "configured" if s.qdrant_api_key else "",
        "llm_provider": s.llm_provider,
        "llm_model": s.llm_model,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "openai_api_key_set": bool(s.openai_api_key),
        "anthropic_api_key_set": bool(s.anthropic_api_key),
        "tavily_api_key_set": bool(s.tavily_api_key),
        "qdrant_api_key_set": bool(s.qdrant_api_key),
    }


@router.post("/api/settings")
def write_settings(payload: SettingsUpdate):
    body = payload.model_dump(exclude_none=True)
    # Touching any Qdrant field forces a backend re-probe so the change
    # actually takes effect on the next query.
    needs_reprobe = "qdrant_url" in body or "qdrant_api_key" in body
    updated = update_settings(body)
    if needs_reprobe:
        reset_backend()
    return {
        "success": True,
        "settings": {
            "llm_provider": updated.llm_provider,
            "llm_model": updated.llm_model,
            "chunk_size": updated.chunk_size,
            "chunk_overlap": updated.chunk_overlap,
            "qdrant_url": updated.qdrant_url,
            "openai_api_key_set": bool(updated.openai_api_key),
            "anthropic_api_key_set": bool(updated.anthropic_api_key),
            "tavily_api_key_set": bool(updated.tavily_api_key),
            "qdrant_api_key_set": bool(updated.qdrant_api_key),
        },
    }


@router.get("/api/system/health")
def health():
    # Force a probe so the badge reflects current Qdrant reachability.
    try:
        get_backend()
    except Exception:
        pass
    name = active_backend_name()
    return {
        "ok": True,
        "vector_backend": name,
        "qdrant_reachable": name == "qdrant",
        "qdrant_active": name == "qdrant",
    }
