"""
Agent service entrypoints — `run_agent` to start, `resume_agent` to continue
from a HITL pause. Both write progress + final output to SQLite so the
FastAPI status endpoint is the single source of truth for the UI.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from backend.agent.graph import get_graph
from backend.db import get_db


async def run_agent(query: str, enable_hitl: bool) -> str:
    db = get_db()
    run_id = str(uuid.uuid4())
    thread_id = run_id  # 1:1 mapping for now
    with db:
        db.execute(
            """INSERT INTO agent_runs (id, query, status, current_step, thread_id)
               VALUES (?, ?, 'running', 'Initializing', ?)""",
            (run_id, query, thread_id),
        )
    # Schedule the workflow as a fire-and-forget task — the API returns the
    # run_id immediately and the client polls /agent/status.
    import asyncio

    asyncio.create_task(_execute(run_id, query, enable_hitl, thread_id))
    return run_id


def _is_interrupted(app, config) -> bool:
    """Check if the graph is paused at an interrupt (HITL)."""
    try:
        state = app.get_state(config)
        # LangGraph sets `state.next` to the tuple of pending nodes
        # when the graph is interrupted. If non-empty, we're paused.
        return bool(state.next)
    except Exception:
        return False


async def _execute(run_id: str, query: str, enable_hitl: bool, thread_id: str) -> None:
    db = get_db()
    start = time.time()

    # ── Pre-flight: validate that the LLM is reachable before running the
    #    full graph.
    try:
        from backend.llm import get_llm
        get_llm()  # will raise RuntimeError if key is missing
    except Exception as e:
        with db:
            db.execute(
                "UPDATE agent_runs SET status='failed', error_log=? WHERE id=?",
                (f"LLM configuration error: {e}", run_id),
            )
        return

    app = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await app.ainvoke(
            {"run_id": run_id, "query": query, "enable_hitl": enable_hitl},
            config=config,
        )
    except GraphInterrupt:
        # Older LangGraph versions raise this on interrupt.
        return
    except Exception as e:
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        with db:
            db.execute(
                "UPDATE agent_runs SET status='failed', error_log=? WHERE id=?",
                (str(e), run_id),
            )
        return

    # ── Check whether the graph paused at an interrupt (HITL) ──────────
    # With MemorySaver, interrupt() does NOT raise GraphInterrupt —
    # ainvoke() returns a partial result. We detect it by checking
    # whether the graph has pending next-nodes.
    if _is_interrupted(app, config):
        # human_input node already set status='awaiting_clarification'
        return

    # ── Normal completion ──────────────────────────────────────────────
    elapsed = time.time() - start
    final = result.get("final_output") or "(no output generated)"
    with db:
        db.execute(
            """UPDATE agent_runs
               SET status='completed', final_output=?, output_format=?,
                   summary=?, execution_time_seconds=?, completed_at=CURRENT_TIMESTAMP
             WHERE id=?""",
            (
                final,
                result.get("output_format"),
                result.get("summary"),
                elapsed,
                run_id,
            ),
        )


async def resume_agent(run_id: str, user_reply: str) -> None:
    db = get_db()
    row = db.execute(
        "SELECT thread_id FROM agent_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if not row:
        raise ValueError("Agent run not found")
    thread_id = row["thread_id"]

    with db:
        db.execute("UPDATE agent_runs SET status = 'running' WHERE id = ?", (run_id,))

    app = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    start = time.time()
    try:
        result = await app.ainvoke(Command(resume=user_reply), config=config)
    except GraphInterrupt:
        return
    except Exception as e:
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        with db:
            db.execute(
                "UPDATE agent_runs SET status='failed', error_log=? WHERE id=?",
                (str(e), run_id),
            )
        return

    # Check for another interrupt (nested HITL — unlikely but robust)
    if _is_interrupted(app, config):
        return

    # ── Normal completion after resume ─────────────────────────────────
    elapsed = time.time() - start
    final = result.get("final_output") or "(no output generated)"
    with db:
        db.execute(
            """UPDATE agent_runs
               SET status='completed', final_output=?, output_format=?,
                   summary=?,
                   execution_time_seconds=COALESCE(execution_time_seconds,0)+?,
                   completed_at=CURRENT_TIMESTAMP
             WHERE id=?""",
            (
                final,
                result.get("output_format"),
                result.get("summary"),
                elapsed,
                run_id,
            ),
        )
