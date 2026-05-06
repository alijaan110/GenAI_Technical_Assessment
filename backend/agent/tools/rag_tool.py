"""
Bridge from the agent into the Task 1 RAG pipeline. Returns retrieved
chunks (not just the answer) so the summarizer can reason over the raw
context and the structured-output node cites sources accurately.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from backend.rag_service import handle_rag_query


async def rag_search(query: str, k: int = 5) -> Dict:
    r = await handle_rag_query(query, top_k=k, use_hybrid=True)
    return {
        "hits": [
            {
                "text": s.excerpt,
                "source": s.document_name,
                "page": s.page,
                "section": s.section,
                "score": s.relevance_score,
            }
            for s in r.sources
        ],
        "answer": r.answer,
        "is_grounded": r.is_grounded,
        "retrieval_score": r.retrieval_score,
    }
