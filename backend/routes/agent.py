"""Agent endpoints — execute, status polling, resume from HITL pause."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.agent.service import resume_agent, run_agent
from backend.db import get_db

router = APIRouter(tags=["agent"])


class ExecuteRequest(BaseModel):
    query: str
    enable_hitl: Optional[bool] = False


class ResumeRequest(BaseModel):
    user_clarification: Optional[str] = ""


@router.post("/api/agent/execute")
async def execute(payload: ExecuteRequest):
    if not payload.query or not payload.query.strip():
        raise HTTPException(400, "query is required")
    run_id = await run_agent(payload.query, bool(payload.enable_hitl))
    return {"success": True, "run_id": run_id}


@router.post("/api/agent/resume/{run_id}")
async def resume(run_id: str, payload: ResumeRequest):
    try:
        await resume_agent(run_id, payload.user_clarification or "")
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"success": True}


@router.get("/api/agent/status/{run_id}")
def status(run_id: str):
    db = get_db()
    run = db.execute("SELECT * FROM agent_runs WHERE id = ?", (run_id,)).fetchone()
    if not run:
        raise HTTPException(404, "Run not found")
    steps_rows = db.execute(
        """SELECT step_name, status, output_data, error_message, started_at, completed_at
           FROM agent_steps WHERE run_id = ? ORDER BY id ASC""",
        (run_id,),
    ).fetchall()
    steps = []
    for s in steps_rows:
        try:
            payload = json.loads(s["output_data"]) if s["output_data"] else None
        except Exception:
            payload = None
        steps.append(
            {
                "step_name": s["step_name"],
                "status": s["status"],
                "result_summary": payload,
                "error": s["error_message"],
                "started_at": s["started_at"],
                "completed_at": s["completed_at"],
            }
        )
    out = dict(run)
    out["steps"] = steps
    return out


@router.get("/api/agent/result/{run_id}")
def result(run_id: str):
    db = get_db()
    row = db.execute(
        """SELECT final_output, output_format, summary, search_strategy,
                  execution_time_seconds, status
           FROM agent_runs WHERE id = ?""",
        (run_id,),
    ).fetchone()
    if not row:
        raise HTTPException(404, "Run not found")
    return dict(row)


@router.delete("/api/agent/runs/{run_id}")
def delete_run(run_id: str):
    db = get_db()
    row = db.execute("SELECT id FROM agent_runs WHERE id = ?", (run_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Run not found")
    with db:
        db.execute("DELETE FROM agent_steps WHERE run_id = ?", (run_id,))
        db.execute("DELETE FROM agent_runs WHERE id = ?", (run_id,))
    return {"success": True}


@router.get("/api/agent/history")
def history():
    db = get_db()
    rows = db.execute(
        """SELECT id, query, status, current_step, output_format, search_strategy,
                  needs_clarification, clarification_question, execution_time_seconds,
                  started_at, completed_at
           FROM agent_runs ORDER BY started_at DESC LIMIT 50"""
    ).fetchall()
    return {"runs": [dict(r) for r in rows]}
