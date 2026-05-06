"""Chat session endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.session_service import (
    create_session,
    delete_session,
    get_messages,
    get_session,
    list_sessions,
    rename_session,
)

router = APIRouter(tags=["sessions"])


class SessionCreate(BaseModel):
    title: Optional[str] = None


class SessionRename(BaseModel):
    title: str


@router.get("/api/sessions")
def get_all():
    return {"sessions": list_sessions()}


@router.post("/api/sessions")
def post_create(payload: SessionCreate):
    return {"session": create_session(payload.title)}


@router.get("/api/sessions/{session_id}")
def get_one(session_id: str):
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    return {"session": sess, "messages": get_messages(session_id)}


@router.patch("/api/sessions/{session_id}")
def patch_rename(session_id: str, payload: SessionRename):
    if not payload.title:
        raise HTTPException(400, "title required")
    rename_session(session_id, payload.title)
    return {"success": True}


@router.delete("/api/sessions/{session_id}")
def del_session(session_id: str):
    delete_session(session_id)
    return {"success": True}
