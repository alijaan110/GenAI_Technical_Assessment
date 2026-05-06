"""
Production RAG query pipeline with token-by-token streaming.

   1. Hybrid retrieve (dense + sparse, RRF-fused).
   2. Pre-retrieval relevance gate — if top RRF score is below threshold,
      treat as out-of-context and refuse without calling the LLM.
   3. Strict system prompt with mandatory inline citations.
   4. Stream tokens from the LLM (LangChain's async astream API).
   5. Post-generation grounded check — flag answers that match refusal
      phrases as not-grounded so the UI can render them differently.
   6. Persist the full answer + sources to queries / query_sources / messages.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from typing import AsyncIterator, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from backend.bm25_store import get_bm25
from backend.chitchat import classify as classify_chitchat, respond as chitchat_respond
from backend.db import get_db
from backend.hybrid_retriever import hybrid_retrieve, HybridHit
from backend.llm import get_llm
from backend.prompts import (
    OUT_OF_CONTEXT_PHRASES,
    OUT_OF_CONTEXT_RESPONSE,
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from backend.settings import get_settings
from backend.vector_store import get_backend


# ── Result shapes ────────────────────────────────────────────────
@dataclass
class RagSource:
    chunk_id: str
    document_name: str
    page: Optional[int]
    section: str
    sub_section: str
    excerpt: str
    relevance_score: float
    dense_score: Optional[float]
    sparse_score: Optional[float]


@dataclass
class RagResult:
    query_id: str
    answer: str
    is_grounded: bool
    response_time: float
    model_used: str
    sources: List[RagSource]
    retrieval_score: float
    session_id: Optional[str] = None


# Dense cosine similarity threshold — if the BEST chunk's dense score
# is below this, the context is genuinely irrelevant and we should refuse
# rather than let the LLM hallucinate from training knowledge.
# 0.30 is conservative: typical relevant hits score 0.40-0.85.
DENSE_RELEVANCE_THRESHOLD = 0.20

# RRF threshold as fallback
RRF_RELEVANCE_THRESHOLD = 0.010


def _is_relevant(hits: List[HybridHit]) -> bool:
    """Check if retrieved context is actually relevant to the query."""
    if not hits:
        return False
    # Primary gate: dense cosine similarity of the best hit
    best_dense = max((h.dense_score or 0.0) for h in hits[:3])
    if best_dense < DENSE_RELEVANCE_THRESHOLD:
        return False
    return hits[0].rrf_score >= RRF_RELEVANCE_THRESHOLD


def _is_grounded(answer: str) -> bool:
    a = answer.lower()
    return not any(p in a for p in OUT_OF_CONTEXT_PHRASES)


def _format_context(hits: List[HybridHit]) -> str:
    parts = []
    for i, h in enumerate(hits):
        m = h.metadata or {}
        doc_type = m.get('doc_type', '')
        jurisdiction = m.get('jurisdiction', '')
        type_info = f", Type: {doc_type}" if doc_type and doc_type != 'document' else ""
        juris_info = f", Jurisdiction: {jurisdiction}" if jurisdiction and jurisdiction != 'Unknown' else ""
        parts.append(
            f"[#{i + 1}] [Source: {m.get('source', 'unknown')}, "
            f"Page {m.get('page', '?')}, Section {m.get('section', 'Unknown')}"
            f"{type_info}{juris_info}]\n"
            f"{h.text.strip()}"
        )
    return "\n\n---\n\n".join(parts)


def _hits_to_sources(hits: List[HybridHit]) -> List[RagSource]:
    out: List[RagSource] = []
    for h in hits:
        m = h.metadata or {}
        excerpt = h.text if len(h.text) <= 320 else h.text[:320] + "…"
        out.append(
            RagSource(
                chunk_id=h.chunk_id,
                document_name=m.get("source", "unknown"),
                page=m.get("page"),
                section=m.get("section", "Unknown"),
                sub_section=m.get("sub_section", ""),
                excerpt=excerpt,
                relevance_score=h.rrf_score,
                dense_score=h.dense_score,
                sparse_score=h.sparse_score,
            )
        )
    return out


def _persist_query(
    query_id: str,
    *,
    session_id: Optional[str],
    question: str,
    answer: str,
    top_k: int,
    use_hybrid: bool,
    is_grounded: bool,
    retrieval_score: float,
    model_used: str,
    response_time: float,
    hits: List[HybridHit],
) -> None:
    db = get_db()
    with db:
        db.execute(
            """INSERT INTO queries
               (id, session_id, question, answer, top_k, use_hybrid,
                is_grounded, retrieval_score, model_used, response_time_seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                query_id,
                session_id,
                question,
                answer,
                top_k,
                1 if use_hybrid else 0,
                1 if is_grounded else 0,
                retrieval_score,
                model_used,
                response_time,
            ),
        )
        for idx, h in enumerate(hits):
            if h.chunk_id:
                db.execute(
                    """INSERT INTO query_sources
                       (query_id, chunk_id, relevance_score, rank_position)
                       VALUES (?, ?, ?, ?)""",
                    (query_id, h.chunk_id, h.rrf_score, idx + 1),
                )


def _append_message(
    session_id: str,
    role: str,
    content: str,
    *,
    sources: List[RagSource] | None = None,
    is_grounded: Optional[bool] = None,
    retrieval_score: Optional[float] = None,
) -> None:
    db = get_db()
    with db:
        db.execute(
            """INSERT INTO chat_messages
               (id, session_id, role, content, sources, is_grounded, retrieval_score)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                session_id,
                role,
                content,
                json.dumps([asdict(s) for s in sources]) if sources else None,
                None if is_grounded is None else (1 if is_grounded else 0),
                retrieval_score,
            ),
        )
        db.execute(
            """UPDATE chat_sessions SET message_count = message_count + 1,
                   updated_at = CURRENT_TIMESTAMP WHERE id = ?""",
            (session_id,),
        )


# ── Non-streaming entrypoint (kept for eval / agent reuse) ──────
async def handle_rag_query(
    question: str,
    *,
    top_k: int = 3,
    use_hybrid: bool = True,
    session_id: Optional[str] = None,
) -> RagResult:
    settings = get_settings()
    query_id = str(uuid.uuid4())

    # Chitchat short-circuit — greetings, identity, thanks etc. never go
    # through retrieval, so we don't get "who are you?" answered with GDPR.
    intent = classify_chitchat(question)
    if intent is not None:
        answer = chitchat_respond(intent)
        _persist_query(
            query_id,
            session_id=session_id,
            question=question,
            answer=answer,
            top_k=top_k,
            use_hybrid=use_hybrid,
            is_grounded=True,  # treat chitchat as grounded — it's an intentional reply
            retrieval_score=0.0,
            model_used="chitchat",
            response_time=0.0,
            hits=[],
        )
        if session_id:
            _append_message(session_id, "user", question)
            _append_message(
                session_id,
                "assistant",
                answer,
                sources=[],
                is_grounded=True,
                retrieval_score=0.0,
            )
        return RagResult(
            query_id=query_id,
            answer=answer,
            is_grounded=True,
            response_time=0.0,
            model_used="chitchat",
            sources=[],
            retrieval_score=0.0,
            session_id=session_id,
        )

    backend = get_backend()
    if backend.size() == 0:
        # Make sure BM25 reflects an empty corpus too.
        get_bm25().rebuild()
        return RagResult(
            query_id=str(uuid.uuid4()),
            answer=(
                "No documents have been ingested yet. Upload at least one PDF in the "
                "Documents panel before asking a question."
            ),
            is_grounded=False,
            response_time=0.0,
            model_used=settings.llm_model or "n/a",
            sources=[],
            retrieval_score=0.0,
            session_id=session_id,
        )

    hits = hybrid_retrieve(question, k=top_k, use_hybrid=use_hybrid)
    top_score = hits[0].rrf_score if hits else 0.0

    if not _is_relevant(hits):
        answer = OUT_OF_CONTEXT_RESPONSE
        _persist_query(
            query_id,
            session_id=session_id,
            question=question,
            answer=answer,
            top_k=top_k,
            use_hybrid=use_hybrid,
            is_grounded=False,
            retrieval_score=top_score,
            model_used=settings.llm_model or "n/a",
            response_time=0.0,
            hits=[],
        )
        if session_id:
            _append_message(session_id, "user", question)
            _append_message(
                session_id,
                "assistant",
                answer,
                sources=[],
                is_grounded=False,
                retrieval_score=top_score,
            )
        return RagResult(
            query_id=query_id,
            answer=answer,
            is_grounded=False,
            response_time=0.0,
            model_used=settings.llm_model or "n/a",
            sources=[],
            retrieval_score=top_score,
            session_id=session_id,
        )

    context = _format_context(hits)
    llm = get_llm(settings)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=USER_PROMPT.format(question=question)),
    ]

    start = time.time()
    response = await llm.ainvoke(messages)
    elapsed = time.time() - start
    answer = response.content if isinstance(response.content, str) else str(response.content)
    grounded = _is_grounded(answer)

    sources = _hits_to_sources(hits)
    _persist_query(
        query_id,
        session_id=session_id,
        question=question,
        answer=answer,
        top_k=top_k,
        use_hybrid=use_hybrid,
        is_grounded=grounded,
        retrieval_score=top_score,
        model_used=settings.llm_model or "n/a",
        response_time=elapsed,
        hits=hits,
    )
    if session_id:
        _append_message(session_id, "user", question)
        _append_message(
            session_id,
            "assistant",
            answer,
            sources=sources,
            is_grounded=grounded,
            retrieval_score=top_score,
        )

    return RagResult(
        query_id=query_id,
        answer=answer,
        is_grounded=grounded,
        response_time=elapsed,
        model_used=settings.llm_model or "n/a",
        sources=sources,
        retrieval_score=top_score,
        session_id=session_id,
    )


# ── Streaming entrypoint (SSE, token-by-token) ──────────────────
async def stream_rag_query(
    question: str,
    *,
    top_k: int = 3,
    use_hybrid: bool = True,
    session_id: Optional[str] = None,
) -> AsyncIterator[Dict]:
    """
    Async generator yielding event dicts:

      {"event": "meta",   "data": {"query_id": ..., "session_id": ...}}
      {"event": "sources","data": [<RagSource>, ...]}
      {"event": "token",  "data": "<chunk text>"}
      {"event": "done",   "data": {"is_grounded": bool, "retrieval_score": float, "response_time": float}}
      {"event": "error",  "data": {"message": "..."}}

    Consumers (the FastAPI route) format these as Server-Sent Events.
    """
    settings = get_settings()
    query_id = str(uuid.uuid4())

    # Chitchat short-circuit (greetings / identity / help / thanks / bye).
    intent = classify_chitchat(question)
    if intent is not None:
        answer = chitchat_respond(intent)
        yield {"event": "meta", "data": {"query_id": query_id, "session_id": session_id}}
        yield {"event": "sources", "data": []}
        for chunk in _stream_static(answer, chunk_chars=24):
            yield {"event": "token", "data": chunk}
        _persist_query(
            query_id,
            session_id=session_id,
            question=question,
            answer=answer,
            top_k=top_k,
            use_hybrid=use_hybrid,
            is_grounded=True,
            retrieval_score=0.0,
            model_used="chitchat",
            response_time=0.0,
            hits=[],
        )
        if session_id:
            _append_message(session_id, "user", question)
            _append_message(
                session_id,
                "assistant",
                answer,
                sources=[],
                is_grounded=True,
                retrieval_score=0.0,
            )
        yield {
            "event": "done",
            "data": {
                "is_grounded": True,
                "retrieval_score": 0.0,
                "response_time": 0.0,
                "query_id": query_id,
            },
        }
        return

    backend = get_backend()
    if backend.size() == 0:
        get_bm25().rebuild()
        msg = (
            "No documents have been ingested yet. Upload at least one PDF in the "
            "Documents panel before asking a question."
        )
        yield {"event": "meta", "data": {"query_id": query_id, "session_id": session_id}}
        yield {"event": "sources", "data": []}
        yield {"event": "token", "data": msg}
        yield {
            "event": "done",
            "data": {"is_grounded": False, "retrieval_score": 0.0, "response_time": 0.0},
        }
        return

    hits = hybrid_retrieve(question, k=top_k, use_hybrid=use_hybrid)
    top_score = hits[0].rrf_score if hits else 0.0
    sources = _hits_to_sources(hits)

    yield {"event": "meta", "data": {"query_id": query_id, "session_id": session_id}}
    yield {"event": "sources", "data": [asdict(s) for s in sources]}

    # Pre-retrieval gate — refuse without calling the LLM.
    if not _is_relevant(hits):
        answer = OUT_OF_CONTEXT_RESPONSE
        # Stream the refusal in small chunks so the UI feels alive.
        for chunk in _stream_static(answer):
            yield {"event": "token", "data": chunk}
        _persist_query(
            query_id,
            session_id=session_id,
            question=question,
            answer=answer,
            top_k=top_k,
            use_hybrid=use_hybrid,
            is_grounded=False,
            retrieval_score=top_score,
            model_used=settings.llm_model or "n/a",
            response_time=0.0,
            hits=[],
        )
        if session_id:
            _append_message(session_id, "user", question)
            _append_message(
                session_id,
                "assistant",
                answer,
                sources=[],
                is_grounded=False,
                retrieval_score=top_score,
            )
        yield {
            "event": "done",
            "data": {
                "is_grounded": False,
                "retrieval_score": top_score,
                "response_time": 0.0,
                "query_id": query_id,
            },
        }
        return

    context = _format_context(hits)
    llm = get_llm(settings)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=USER_PROMPT.format(question=question)),
    ]

    start = time.time()
    accumulated: List[str] = []
    try:
        async for chunk in llm.astream(messages):
            piece = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            if not piece:
                continue
            accumulated.append(piece)
            yield {"event": "token", "data": piece}
    except Exception as e:
        yield {"event": "error", "data": {"message": str(e)}}
        return

    elapsed = time.time() - start
    answer = "".join(accumulated)
    grounded = _is_grounded(answer)

    _persist_query(
        query_id,
        session_id=session_id,
        question=question,
        answer=answer,
        top_k=top_k,
        use_hybrid=use_hybrid,
        is_grounded=grounded,
        retrieval_score=top_score,
        model_used=settings.llm_model or "n/a",
        response_time=elapsed,
        hits=hits,
    )
    if session_id:
        _append_message(session_id, "user", question)
        _append_message(
            session_id,
            "assistant",
            answer,
            sources=sources,
            is_grounded=grounded,
            retrieval_score=top_score,
        )

    yield {
        "event": "done",
        "data": {
            "is_grounded": grounded,
            "retrieval_score": top_score,
            "response_time": elapsed,
            "query_id": query_id,
        },
    }


def _stream_static(text: str, chunk_chars: int = 32):
    """Chunk a static string so the OOC refusal still 'types out'."""
    for i in range(0, len(text), chunk_chars):
        yield text[i : i + chunk_chars]
