"""
Summarizer tool — collapses retrieved RAG chunks + web hits into a tight,
400–600 word brief focused on the original query. Feeds the structured-output
node which then formats the final checklist or report.
"""

from __future__ import annotations

from typing import Dict, List

from langchain_core.messages import HumanMessage

from backend.llm import get_llm

SUMMARIZE_PROMPT = """You are a senior legal research analyst. Produce a focused 400-600 word
research brief on the user's QUERY using the provided INTERNAL DOCUMENTS and
WEB FINDINGS.

Rules:
- Distinguish between facts from internal documents (cite as [Source: file, p.X]) and web findings (cite as [Web: domain]).
- Focus on: regulations, requirements, obligations, deadlines, penalties.
- Surface contradictions or gaps explicitly.
- Do NOT invent facts that aren't in the inputs.

QUERY: {query}

INTERNAL DOCUMENTS:
{rag}

WEB FINDINGS:
{web}

Produce the brief now."""


async def summarize(*, query: str, rag: List[Dict], web: List[Dict]) -> str:
    if not rag and not web:
        return "(no inputs to summarize)"
    rag_text = "\n---\n".join(
        f"[Source: {r.get('source','?')}, p.{r.get('page','?')}, {r.get('section','?')}]\n{r.get('text','')}"
        for r in rag
    ) or "(no internal documents retrieved)"
    web_text = "\n---\n".join(
        f"[Web: {w.get('title','')}] ({w.get('url','')})\n{(w.get('text','') or '')[:800]}"
        for w in web
    ) or "(no web findings — search disabled or returned nothing)"

    llm = get_llm()
    prompt = SUMMARIZE_PROMPT.format(
        query=query, rag=rag_text[:6000], web=web_text[:6000]
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content if isinstance(response.content, str) else str(response.content)
