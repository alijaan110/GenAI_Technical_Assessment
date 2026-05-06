"""
Settings — Pydantic-typed, env-aware, persistable to SQLite for runtime
updates from the UI. Environment variables always win over DB-stored
values, so deployments stay declarative.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from backend.db import get_db

load_dotenv()


class AppSettings(BaseModel):
    openai_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")
    tavily_api_key: str = Field(default="")
    qdrant_url: str = Field(default="")
    qdrant_api_key: str = Field(default="")
    llm_provider: str = Field(default="openai")
    llm_model: str = Field(default="gpt-4o-mini")
    # Tuned for legal text: 900-char windows preserve enough surrounding
    # clause context for accurate citations without diluting the embedding's
    # signal-to-noise ratio. Overlap keeps statements that span chunk
    # boundaries findable from either side.
    chunk_size: str = Field(default="900")
    chunk_overlap: str = Field(default="180")


_DEFAULTS = AppSettings(
    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
    tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
    qdrant_url=os.getenv("QDRANT_URL", ""),
    qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
    llm_provider=os.getenv("LLM_PROVIDER", "openai"),
    llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
)

_cached: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    global _cached
    if _cached is not None:
        return _cached

    db = get_db()
    rows = db.execute("SELECT key, value FROM settings").fetchall()
    merged = _DEFAULTS.model_dump()
    for r in rows:
        if r["key"] in merged:
            merged[r["key"]] = r["value"]

    # Env-vars override stored values for any non-empty assignments
    # — keeps secret rotation declarative across redeploys.
    for env_key, slot in [
        ("OPENAI_API_KEY", "openai_api_key"),
        ("ANTHROPIC_API_KEY", "anthropic_api_key"),
        ("TAVILY_API_KEY", "tavily_api_key"),
        ("QDRANT_URL", "qdrant_url"),
        ("QDRANT_API_KEY", "qdrant_api_key"),
    ]:
        v = os.getenv(env_key)
        if v:
            merged[slot] = v

    _cached = AppSettings(**merged)
    return _cached


def update_settings(updates: dict) -> AppSettings:
    db = get_db()
    with db:
        for k, v in updates.items():
            if v is None:
                continue
            db.execute(
                """INSERT INTO settings(key, value, updated_at)
                   VALUES(?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(key) DO UPDATE SET
                       value = excluded.value,
                       updated_at = CURRENT_TIMESTAMP""",
                (k, str(v)),
            )
    invalidate_cache()
    return get_settings()


def invalidate_cache() -> None:
    global _cached
    _cached = None
