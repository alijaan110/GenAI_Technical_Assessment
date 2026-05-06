"""
RAG endpoints — both non-streaming JSON (for the eval/agent reuse path) and
the streaming SSE endpoint that powers the chat UI's token-by-token output.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.rag_service import handle_rag_query, stream_rag_query
from backend.session_service import maybe_auto_title

router = APIRouter(tags=["rag"])


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 8
    use_hybrid: Optional[bool] = True
    session_id: Optional[str] = None


@router.post("/api/rag/query")
async def rag_query(req: QueryRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(400, "question is required")
    result = await handle_rag_query(
        req.question,
        top_k=req.top_k or 5,
        use_hybrid=bool(req.use_hybrid if req.use_hybrid is not None else True),
        session_id=req.session_id,
    )
    if req.session_id:
        maybe_auto_title(req.session_id, req.question)
    return {
        "query_id": result.query_id,
        "answer": result.answer,
        "is_grounded": result.is_grounded,
        "response_time": result.response_time,
        "model_used": result.model_used,
        "retrieval_score": result.retrieval_score,
        "session_id": result.session_id,
        "sources": [asdict(s) for s in result.sources],
    }


@router.post("/api/rag/stream")
async def rag_stream(req: QueryRequest, request: Request):
    """
    Server-Sent Events endpoint. Emits:
      event: meta    data: {"query_id": ..., "session_id": ...}
      event: sources data: [...]
      event: token   data: "<chunk>"
      event: done    data: {...}
    """
    if not req.question or not req.question.strip():
        raise HTTPException(400, "question is required")

    if req.session_id:
        maybe_auto_title(req.session_id, req.question)

    async def event_publisher():
        async for evt in stream_rag_query(
            req.question,
            top_k=req.top_k or 5,
            use_hybrid=bool(req.use_hybrid if req.use_hybrid is not None else True),
            session_id=req.session_id,
        ):
            # If the client disconnects mid-stream, sse_starlette will
            # raise inside the generator on the next yield — exit cleanly.
            if await request.is_disconnected():
                break
            yield {"event": evt["event"], "data": json.dumps(evt["data"])}

    # ping every 15s prevents corporate proxies from killing the SSE stream.
    return EventSourceResponse(event_publisher(), ping=15)
