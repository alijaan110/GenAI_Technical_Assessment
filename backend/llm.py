"""
LLM factory — returns a streaming-capable LangChain ChatModel for the
configured provider.

Cache-bypass note:
  We invalidate the settings cache on every call before reading. The cost is
  a single SQLite SELECT (microseconds); the payoff is that a freshly-saved
  API key in /api/settings is honoured by the *next* LLM invocation rather
  than the next process restart. Without this, an agent run launched from a
  stale Python worker would fail with "Missing credentials" even though the
  user just typed the key into Settings 2 seconds ago.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from backend.settings import AppSettings, get_settings, invalidate_cache


_KEY_HELP = (
    " — paste it in the Settings tab of the LexAI UI, "
    "or set the corresponding environment variable before starting the server."
)


def get_llm(
    settings: AppSettings | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
):
    if settings is None:
        invalidate_cache()
        settings = get_settings()
    s = settings
    provider = (s.llm_provider or "openai").strip().lower()
    model = s.llm_model or ("gpt-4o-mini" if provider == "openai" else "claude-3-5-sonnet-20241022")

    if provider == "openai":
        if not s.openai_api_key or not s.openai_api_key.strip():
            raise RuntimeError("OpenAI API key not configured" + _KEY_HELP)
        return ChatOpenAI(
            api_key=s.openai_api_key.strip(),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
        )

    if provider == "anthropic":
        if not s.anthropic_api_key or not s.anthropic_api_key.strip():
            raise RuntimeError("Anthropic API key not configured" + _KEY_HELP)
        return ChatAnthropic(
            api_key=s.anthropic_api_key.strip(),
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
        )

    raise RuntimeError(f"Unsupported LLM provider: {provider!r}")
