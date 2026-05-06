"""
Tavily-powered web search with graceful fallback.

Tavily ships a Python client; we still hit the REST endpoint directly to
avoid the extra dependency surface. If no key is configured (or the call
fails) we degrade to an empty list so the rest of the agent completes —
the assessment doc explicitly recommends graceful degradation here.
"""

from __future__ import annotations

from typing import Dict, List

import httpx

from backend.settings import get_settings

TAVILY_ENDPOINT = "https://api.tavily.com/search"


async def web_search(query: str, max_results: int = 5) -> List[Dict]:
    settings = get_settings()
    api_key = settings.tavily_api_key
    if not api_key:
        print("[web_search] No TAVILY_API_KEY — skipping live web search.")
        return []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                TAVILY_ENDPOINT,
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "include_answer": False,
                    "max_results": max_results,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", []) or []
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "text": r.get("content", "") or r.get("snippet", ""),
                    "score": float(r.get("score", 0.0) or 0.0),
                }
                for r in results[:max_results]
            ]
    except Exception as e:
        print(f"[web_search] failed: {e}")
        return []
