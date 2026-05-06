"""
LLM-based reranker — scores each retrieved chunk's relevance to the query
and filters out irrelevant noise before the generator sees it.

This is a precision-oriented step: it discards chunks that passed the
coarse RRF gate but don't actually help answer the question. The result
is a tighter, higher-signal context window → fewer hallucinations and
higher faithfulness.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import List, Optional

from langchain_core.messages import HumanMessage

from backend.hybrid_retriever import HybridHit
from backend.llm import get_llm
from backend.settings import get_settings

RERANK_PROMPT = """Rate how relevant this PASSAGE is for answering the QUESTION.
Score from 0 (completely irrelevant) to 10 (directly answers the question).

QUESTION: {question}

PASSAGE:
{passage}

Return ONLY a JSON object: {{"score": <0-10>, "reason": "<one sentence>"}}"""

# Chunks scoring below this threshold are discarded.
RELEVANCE_THRESHOLD = 3
# Minimum number of chunks to keep even if all score low.
MIN_CHUNKS = 3


async def _score_one(llm, question: str, hit: HybridHit) -> tuple[HybridHit, float]:
    """Score a single chunk's relevance to the question."""
    prompt = RERANK_PROMPT.format(
        question=question,
        passage=hit.text[:1500],  # cap to avoid token overflow
    )
    try:
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        # Parse the score from JSON
        cleaned = raw.replace("```json", "").replace("```", "")
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end >= 0:
            obj = json.loads(cleaned[start : end + 1])
            score = float(obj.get("score", 0))
        else:
            m = re.search(r"\b(\d+(?:\.\d+)?)\b", raw)
            score = float(m.group(1)) if m else 0.0
        return hit, max(0.0, min(10.0, score))
    except Exception:
        # On error, keep the chunk with a neutral score
        return hit, 5.0


async def rerank(
    question: str,
    hits: List[HybridHit],
    *,
    threshold: float = RELEVANCE_THRESHOLD,
    min_keep: int = MIN_CHUNKS,
) -> List[HybridHit]:
    """
    Rerank hits by LLM-judged relevance. Returns a filtered, re-sorted list.
    
    The reranker uses a cheap, fast LLM call per chunk. With 10 chunks this
    adds ~1-2s of latency but dramatically improves precision.
    """
    if len(hits) <= min_keep:
        return hits

    settings = get_settings()
    llm = get_llm(settings, temperature=0.0, max_tokens=100)

    tasks = [_score_one(llm, question, h) for h in hits]
    results = await asyncio.gather(*tasks)

    # Sort by reranker score descending
    scored = sorted(results, key=lambda x: x[1], reverse=True)

    # Keep chunks above threshold, but always keep at least min_keep
    kept = [(h, s) for h, s in scored if s >= threshold]
    if len(kept) < min_keep:
        kept = list(scored[:min_keep])

    return [h for h, _ in kept]
