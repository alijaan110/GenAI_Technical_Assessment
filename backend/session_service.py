"""
Chat sessions — backend half of the ChatGPT-style sidebar.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from backend.db import get_db


def _row_to_session(r) -> Dict:
    return {
        "id": r["id"],
        "title": r["title"],
        "message_count": r["message_count"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    }


def create_session(title: Optional[str] = None) -> Dict:
    db = get_db()
    sid = str(uuid.uuid4())
    final_title = (title or "New chat")[:200]
    with db:
        db.execute(
            "INSERT INTO chat_sessions(id, title, message_count) VALUES(?, ?, 0)",
            (sid, final_title),
        )
    return get_session(sid)  # type: ignore[return-value]


def list_sessions() -> List[Dict]:
    db = get_db()
    rows = db.execute(
        """SELECT id, title, message_count, created_at, updated_at
           FROM chat_sessions ORDER BY updated_at DESC LIMIT 200"""
    ).fetchall()
    return [_row_to_session(r) for r in rows]


def get_session(session_id: str) -> Optional[Dict]:
    db = get_db()
    r = db.execute(
        """SELECT id, title, message_count, created_at, updated_at
           FROM chat_sessions WHERE id = ?""",
        (session_id,),
    ).fetchone()
    return _row_to_session(r) if r else None


def delete_session(session_id: str) -> None:
    db = get_db()
    with db:
        db.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        db.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))


def rename_session(session_id: str, title: str) -> None:
    db = get_db()
    with db:
        db.execute(
            """UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (title[:200], session_id),
        )


def get_messages(session_id: str) -> List[Dict[str, Any]]:
    db = get_db()
    # ORDER BY rowid (SQLite's internal monotonic insert counter) — immune
    # to second-precision timestamp ties between the user and assistant
    # messages of a single turn, which previously caused the assistant
    # bubble to occasionally render above its question on reload.
    rows = db.execute(
        """SELECT id, session_id, role, content, sources, is_grounded,
                  retrieval_score, created_at
           FROM chat_messages WHERE session_id = ?
           ORDER BY rowid ASC""",
        (session_id,),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        sources_raw = r["sources"]
        try:
            sources = json.loads(sources_raw) if sources_raw else None
        except Exception:
            sources = None
        out.append(
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "role": r["role"],
                "content": r["content"],
                "sources": sources,
                "is_grounded": r["is_grounded"],
                "retrieval_score": r["retrieval_score"],
                "created_at": r["created_at"],
            }
        )
    return out


def maybe_auto_title(session_id: str, first_user_question: str) -> None:
    """Auto-name a brand-new session from the first user prompt."""
    sess = get_session(session_id)
    if not sess:
        return
    if sess["title"] and sess["title"] != "New chat":
        return
    t = " ".join(first_user_question.strip().split())
    if len(t) > 60:
        t = t[:60].rsplit(" ", 1)[0] + "…"
    rename_session(session_id, t or "New chat")
