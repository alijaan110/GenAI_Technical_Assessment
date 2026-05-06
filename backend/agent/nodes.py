"""
LangGraph node functions. Each node returns the partial state it wants to
merge — LangGraph applies the channel reducers we declared in state.py.

Per-node DB writes (agent_steps) keep the audit trail outside the graph
state so the UI can render live progress without subscribing to the
checkpointer.
"""

from __future__ import annotations

import json
from typing import Dict

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from backend.agent.state import AgentState
from backend.agent.tools.rag_tool import rag_search
from backend.agent.tools.structured_output_tool import structured_output
from backend.agent.tools.summarizer_tool import summarize
from backend.agent.tools.web_search_tool import web_search
from backend.db import get_db
from backend.llm import get_llm


# ── Step persistence helpers ─────────────────────────────────────
def _start_step(run_id: str, name: str, payload: Dict | None = None) -> None:
    db = get_db()
    with db:
        db.execute(
            """INSERT INTO agent_steps
               (run_id, step_name, status, output_data, started_at)
               VALUES (?, ?, 'running', ?, CURRENT_TIMESTAMP)""",
            (run_id, name, json.dumps(payload) if payload else None),
        )
        db.execute("UPDATE agent_runs SET current_step = ? WHERE id = ?", (name, run_id))


def _finish_step(
    run_id: str,
    name: str,
    *,
    status: str = "completed",
    payload: Dict | None = None,
    error: str | None = None,
) -> None:
    db = get_db()
    with db:
        db.execute(
            """UPDATE agent_steps
               SET status = ?, output_data = ?, error_message = ?,
                   completed_at = CURRENT_TIMESTAMP
               WHERE rowid = (
                   SELECT rowid FROM agent_steps
                   WHERE run_id = ? AND step_name = ?
                   ORDER BY rowid DESC LIMIT 1
               )""",
            (status, json.dumps(payload) if payload else None, error, run_id, name),
        )


# ── Nodes ────────────────────────────────────────────────────────
ANALYZER_PROMPT = """Analyze this legal research query and produce a JSON plan:

QUERY: "{query}"

Decide:
- needs_clarification: true ONLY if the query is genuinely ambiguous (vague entity, multiple plausible interpretations). Default false.
- clarification_question: if needs_clarification true, the single question to ask the user. Otherwise empty string.
- output_format: one of "checklist" (compliance / step-by-step), "report" (analysis / explanation), "summary" (very short).
- search_strategy: one of "rag_only", "web_only", "both". Use "both" by default; "rag_only" for strictly internal questions; "web_only" for time-sensitive news.

Return ONLY valid JSON:
{{"needs_clarification": <bool>, "clarification_question": "<string>", "output_format": "<checklist|report|summary>", "search_strategy": "<rag_only|web_only|both>"}}"""


async def query_analyzer(state: AgentState) -> Dict:
    run_id = state["run_id"]
    _start_step(run_id, "Query Analysis")
    try:
        llm = get_llm()
        resp = await llm.ainvoke(
            [HumanMessage(content=ANALYZER_PROMPT.format(query=state["query"]))]
        )
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        s, e = cleaned.find("{"), cleaned.rfind("}")
        parsed = json.loads(cleaned[s : e + 1]) if s >= 0 and e >= 0 else {}

        needs = bool(parsed.get("needs_clarification")) and state.get("enable_hitl", False)
        clar_q = (
            str(parsed.get("clarification_question", "")).strip()
            if needs
            else ""
        )
        if needs and not clar_q:
            clar_q = "Could you clarify your request?"

        fmt = parsed.get("output_format", "report")
        if fmt not in ("checklist", "report", "summary"):
            fmt = "report"
        strat = parsed.get("search_strategy", "both")
        if strat not in ("rag_only", "web_only", "both"):
            strat = "both"

        # Persist analyzer decision so the UI can render it before the next step.
        db = get_db()
        with db:
            db.execute(
                """UPDATE agent_runs
                   SET output_format = ?, search_strategy = ?,
                       needs_clarification = ?, clarification_question = ?
                 WHERE id = ?""",
                (fmt, strat, 1 if needs else 0, clar_q or None, run_id),
            )

        _finish_step(
            run_id,
            "Query Analysis",
            payload={
                "output_format": fmt,
                "search_strategy": strat,
                "needs_clarification": needs,
                "clarification_question": clar_q or None,
            },
        )
        return {
            "needs_clarification": needs,
            "clarification_question": clar_q,
            "output_format": fmt,
            "search_strategy": strat,
            "step_log": [f"🔍 Query analyzed: format={fmt}, strategy={strat}"],
        }
    except Exception as e:
        _finish_step(run_id, "Query Analysis", status="failed", error=str(e))
        return {
            "needs_clarification": False,
            "output_format": "report",
            "search_strategy": "both",
            "error_log": [f"Query analyzer failed: {e}"],
            "step_log": ["⚠️ Query analyzer failed — defaulting to report/both"],
        }


async def human_input(state: AgentState) -> Dict:
    """Pauses the graph via LangGraph's interrupt() and surfaces the
    pending clarification question to the API/UI."""
    run_id = state["run_id"]
    db = get_db()
    with db:
        db.execute(
            "UPDATE agent_runs SET status = 'awaiting_clarification' WHERE id = ?",
            (run_id,),
        )
    _start_step(run_id, "Human Input", {"clarification_question": state["clarification_question"]})

    user_reply = interrupt(
        {
            "question": state.get("clarification_question", ""),
            "run_id": run_id,
        }
    )
    user_reply = str(user_reply) if user_reply is not None else ""

    with db:
        db.execute(
            "UPDATE agent_runs SET status = 'running', user_clarification = ? WHERE id = ?",
            (user_reply, run_id),
        )
    _finish_step(run_id, "Human Input", payload={"user_clarification": user_reply})

    refined = (
        f"{state['query']}\n\nUser clarification: {user_reply}"
        if user_reply.strip()
        else state["query"]
    )
    return {
        "user_clarification": user_reply,
        "query": refined,
        "step_log": [f"🙋 Received clarification: {user_reply or '(empty)'}"],
    }


async def rag_search_node(state: AgentState) -> Dict:
    run_id = state["run_id"]
    if state.get("search_strategy") == "web_only":
        _start_step(run_id, "RAG Search")
        _finish_step(run_id, "RAG Search", payload={"skipped": "web_only mode"})
        return {"rag_hits": [], "step_log": ["⏭ RAG skipped (web_only)"]}
    _start_step(run_id, "RAG Search")
    try:
        r = await rag_search(state["query"], k=5)
        _finish_step(
            run_id,
            "RAG Search",
            payload={
                "hits": len(r["hits"]),
                "grounded": r["is_grounded"],
                "top_score": r["retrieval_score"],
            },
        )
        return {
            "rag_hits": r["hits"],
            "step_log": [
                f"📚 RAG search: {len(r['hits'])} chunks (grounded={r['is_grounded']})"
            ],
        }
    except Exception as e:
        _finish_step(run_id, "RAG Search", status="failed", error=str(e))
        return {
            "rag_hits": [],
            "error_log": [f"RAG search failed: {e}"],
            "step_log": ["⚠️ RAG search failed — continuing with web only"],
        }


async def web_search_node(state: AgentState) -> Dict:
    run_id = state["run_id"]
    if state.get("search_strategy") == "rag_only":
        _start_step(run_id, "Web Search")
        _finish_step(run_id, "Web Search", payload={"skipped": "rag_only mode"})
        return {"web_hits": [], "step_log": ["⏭ Web search skipped (rag_only)"]}
    _start_step(run_id, "Web Search")
    try:
        hits = await web_search(state["query"], max_results=5)
        _finish_step(
            run_id,
            "Web Search",
            payload={"hits": len(hits), "top_url": hits[0]["url"] if hits else None},
        )
        return {
            "web_hits": hits,
            "step_log": [f"🌐 Web search: {len(hits)} results"],
        }
    except Exception as e:
        _finish_step(run_id, "Web Search", status="failed", error=str(e))
        return {
            "web_hits": [],
            "error_log": [f"Web search failed: {e}"],
            "step_log": ["⚠️ Web search failed — continuing with RAG only"],
        }


async def summarizer_node(state: AgentState) -> Dict:
    run_id = state["run_id"]
    rag_hits = state.get("rag_hits", []) or []
    web_hits = state.get("web_hits", []) or []
    if not rag_hits and not web_hits:
        msg = "No content retrieved from RAG or web — cannot summarize."
        _start_step(run_id, "Summarization")
        _finish_step(run_id, "Summarization", status="failed", error=msg)
        return {
            "summary": msg,
            "error_log": [msg],
            "step_log": ["⚠️ Summarization aborted — no inputs"],
        }
    _start_step(run_id, "Summarization")
    try:
        summary = await summarize(query=state["query"], rag=rag_hits, web=web_hits)
        db = get_db()
        with db:
            db.execute("UPDATE agent_runs SET summary = ? WHERE id = ?", (summary, run_id))
        _finish_step(run_id, "Summarization", payload={"length": len(summary)})
        return {
            "summary": summary,
            "step_log": [f"📝 Summarized to ~{len(summary.split())} words"],
        }
    except Exception as e:
        _finish_step(run_id, "Summarization", status="failed", error=str(e))
        return {
            "summary": "Summarization failed.",
            "error_log": [f"Summarizer failed: {e}"],
            "step_log": ["⚠️ Summarizer failed"],
        }


async def structured_output_node(state: AgentState) -> Dict:
    run_id = state["run_id"]
    _start_step(run_id, "Structured Output")
    try:
        out = await structured_output(
            query=state["query"],
            summary=state.get("summary", ""),
            output_format=state.get("output_format", "report"),
        )
        _finish_step(
            run_id,
            "Structured Output",
            payload={"format": state.get("output_format"), "length": len(out)},
        )
        return {
            "final_output": out,
            "step_log": [f"✅ {state.get('output_format','report')} generated"],
        }
    except Exception as e:
        _finish_step(run_id, "Structured Output", status="failed", error=str(e))
        return {
            "final_output": f"Failed to generate structured output: {e}",
            "error_log": [f"Structured output failed: {e}"],
            "step_log": ["⚠️ Structured output failed"],
        }
