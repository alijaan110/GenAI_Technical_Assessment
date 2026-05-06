"""
LangGraph state schema. Each field uses a default factory so partial updates
returned by node functions only overwrite their specific channel.
"""

from __future__ import annotations

import operator
from typing import Annotated, Dict, List, Literal, Optional, TypedDict


class AgentState(TypedDict, total=False):
    run_id: str
    query: str

    # Analyzer outputs
    needs_clarification: bool
    clarification_question: str
    user_clarification: str
    output_format: Literal["checklist", "report", "summary"]
    search_strategy: Literal["rag_only", "web_only", "both"]
    enable_hitl: bool

    # Research data
    rag_hits: List[Dict]
    web_hits: List[Dict]
    summary: str

    # Output
    final_output: str

    # Telemetry — accumulated across nodes via operator.add
    error_log: Annotated[List[str], operator.add]
    step_log: Annotated[List[str], operator.add]
