"""
Auto-generate gold-standard Q&A pairs from ingested chunks (Method A in the
assessment doc). The LLM is shown a chunk and asked to produce a mix of
factual / conceptual / cross-reference / edge-case questions.
"""

from __future__ import annotations

import json
import uuid
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage

from backend.db import get_db
from backend.llm import get_llm

GENERATION_PROMPT = """You are creating a gold-standard test dataset for a legal RAG system.
Given the document excerpt, generate {n} diverse, high-quality test questions.

Include a mix of:
- Specific factual questions (names, dates, numbers, article references)
- Conceptual questions (what does X mean?)
- Cross-reference questions (how does A relate to B?)
- Edge case questions (what happens when...?)

Each question must be answerable strictly from the excerpt. Do not invent facts.

DOCUMENT NAME: {source}
PAGE: {page}
SECTION: {section}

EXCERPT:
\"\"\"
{context}
\"\"\"

Return ONLY valid JSON in the form:
{{"questions": [
  {{"question": "...", "ground_truth": "...", "question_type": "factual|conceptual|cross_reference|edge_case"}}
]}}"""


def _parse_questions(raw: str) -> List[Dict]:
    try:
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s < 0 or e < 0:
            return []
        obj = json.loads(cleaned[s : e + 1])
        arr = obj.get("questions", [])
        if not isinstance(arr, list):
            return []
        out: List[Dict] = []
        for q in arr:
            if not q.get("question") or not q.get("ground_truth"):
                continue
            out.append(
                {
                    "question": str(q["question"]).strip(),
                    "ground_truth": str(q["ground_truth"]).strip(),
                    "question_type": str(q.get("question_type", "factual")).lower(),
                }
            )
        return out
    except Exception as e:
        print(f"[dataset_generator] parse failed: {e}")
        return []


async def auto_generate_test_set(
    *,
    n_per_chunk: int = 2,
    max_chunks: int = 10,
    document_id: Optional[str] = None,
) -> Dict:
    n_per_chunk = max(1, min(5, n_per_chunk))
    max_chunks = max(1, min(50, max_chunks))
    db = get_db()
    llm = get_llm()

    if document_id:
        rows = db.execute(
            """SELECT id, chunk_text, page_number, section_title, metadata
               FROM chunks WHERE document_id = ? AND char_count >= 350
               ORDER BY RANDOM() LIMIT ?""",
            (document_id, max_chunks),
        ).fetchall()
    else:
        rows = db.execute(
            """SELECT id, chunk_text, page_number, section_title, metadata
               FROM chunks WHERE char_count >= 350
               ORDER BY RANDOM() LIMIT ?""",
            (max_chunks,),
        ).fetchall()

    if not rows:
        raise ValueError(
            "No suitable chunks available for test generation — upload at least one document first."
        )

    all_qs: List[Dict] = []
    for r in rows:
        meta = json.loads(r["metadata"] or "{}")
        prompt = GENERATION_PROMPT.format(
            n=n_per_chunk,
            source=meta.get("source", "document"),
            page=r["page_number"] or meta.get("page", "?"),
            section=r["section_title"] or meta.get("section", "Unknown"),
            context=r["chunk_text"],
        )
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        for q in _parse_questions(raw):
            q["source_doc"] = meta.get("source")
            q["source_page"] = r["page_number"] or meta.get("page")
            all_qs.append(q)

    test_set_id = str(uuid.uuid4())
    with db:
        for q in all_qs:
            db.execute(
                """INSERT INTO test_questions
                   (id, test_set_id, question, expected_answer, question_type, source_doc, source_page)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    test_set_id,
                    q["question"],
                    q["ground_truth"],
                    q["question_type"],
                    q.get("source_doc"),
                    q.get("source_page"),
                ),
            )
    return {"test_set_id": test_set_id, "question_count": len(all_qs)}
