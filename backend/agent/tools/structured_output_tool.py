"""
Final-output formatter. Produces one of:
  - Compliance checklist (Markdown checkboxes by category)
  - Research report (Executive summary, Findings, Recommendations)
  - Concise summary
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from backend.llm import get_llm

CHECKLIST_PROMPT = """Based on the BRIEF below, create a structured Compliance Checklist for the QUERY.

Use exactly this Markdown format:

## Compliance Checklist: <topic>

### Category 1: <name>
- [ ] <Requirement> — *Legal basis: <Article / Source>*
- [ ] <Requirement> — *Deadline: <date or ongoing>*

### Category 2: <name>
- [ ] ...

### Penalties for Non-Compliance
- ...

### Recommended Next Steps
1. ...

QUERY: {query}

BRIEF:
{summary}

Generate the checklist now."""

REPORT_PROMPT = """Based on the BRIEF below, produce a detailed legal research REPORT for the QUERY.

Use exactly this Markdown format:

# Legal Research Report: <topic>

## Executive Summary
<2-3 sentences>

## Key Findings
- <finding> — *Source: <citation>*
- <finding> — *Source: <citation>*

## Detailed Analysis
### <Subtopic 1>
<paragraph>

### <Subtopic 2>
<paragraph>

## Risks & Open Questions
- ...

## Recommendations
1. ...

QUERY: {query}

BRIEF:
{summary}

Generate the report now."""

SUMMARY_PROMPT = """Polish the BRIEF below into a tight executive summary (max 200 words) for the QUERY.

QUERY: {query}

BRIEF:
{summary}

Produce the summary now."""


async def structured_output(*, query: str, summary: str, output_format: str) -> str:
    template = (
        CHECKLIST_PROMPT
        if output_format == "checklist"
        else SUMMARY_PROMPT
        if output_format == "summary"
        else REPORT_PROMPT
    )
    prompt = template.format(query=query, summary=summary)
    llm = get_llm()
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content if isinstance(response.content, str) else str(response.content)
