"""
LangGraph assembly.

  START
    │
    ▼
  query_analyzer ─── classifies query, picks output_format,
    │                  decides whether clarification is needed
    ├── (needs_clarification = true) ──► human_input ──► (resumes after user reply)
    │                                          │
    ▼                                          ▼
  rag_search ─────────────────────────► web_search ───► summarizer ───► structured_output ───► END

State persistence:
  - LangGraph MemorySaver for graph-state checkpointing (in-process,
    sufficient for the assessment).
  - SQLite agent_runs / agent_steps tables for the audit trail and the
    UI's live progress polling.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from backend.agent.nodes import (
    human_input,
    query_analyzer,
    rag_search_node,
    structured_output_node,
    summarizer_node,
    web_search_node,
)
from backend.agent.state import AgentState

_compiled = None


def _should_clarify(state: AgentState) -> str:
    return "human_input" if state.get("needs_clarification") else "rag_search"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("query_analyzer", query_analyzer)
    g.add_node("human_input", human_input)
    g.add_node("rag_search", rag_search_node)
    g.add_node("web_search", web_search_node)
    g.add_node("summarizer", summarizer_node)
    g.add_node("structured_output", structured_output_node)

    g.add_edge(START, "query_analyzer")
    g.add_conditional_edges(
        "query_analyzer",
        _should_clarify,
        {"human_input": "human_input", "rag_search": "rag_search"},
    )
    g.add_edge("human_input", "rag_search")
    g.add_edge("rag_search", "web_search")
    g.add_edge("web_search", "summarizer")
    g.add_edge("summarizer", "structured_output")
    g.add_edge("structured_output", END)

    return g.compile(checkpointer=MemorySaver())


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled
