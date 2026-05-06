"""
Embedding factory — wraps OpenAI text-embedding-3-small (1536 dims).
Returns the LangChain object so it composes with LangChain stores; the
raw OpenAI client is exposed for batch embedding when we need it.
"""

from __future__ import annotations

from typing import List

from langchain_openai import OpenAIEmbeddings

from backend.settings import get_settings

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


def build_embeddings() -> OpenAIEmbeddings:
    s = get_settings()
    if not s.openai_api_key:
        raise RuntimeError(
            "OpenAI API key not configured. Set it in Settings before "
            "uploading documents or querying."
        )
    return OpenAIEmbeddings(api_key=s.openai_api_key, model=EMBED_MODEL)


def embed_documents(texts: List[str]) -> List[List[float]]:
    return build_embeddings().embed_documents(texts)


def embed_query(text: str) -> List[float]:
    return build_embeddings().embed_query(text)
