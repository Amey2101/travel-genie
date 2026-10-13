"""Explicit LLM provider selection for the requirement-extraction boundary.

Selection comes from ``LLM_PROVIDER`` (``stub`` | ``gemini`` | ``groq``) and
is applied at wiring time. There is deliberately **no automatic fallback**: the
selected provider is used for every extraction call, so a failing provider is
observable instead of silently replaced. ``stub`` requires no credentials.
"""

from __future__ import annotations

from app.core import config
from app.providers.gemini_llm import GeminiLLMClient
from app.providers.groq_llm import GroqLLMClient
from app.providers.llm import LLMClient, LLMExtractionOutput
from app.providers.stub_llm import StubLLMClient


def build_llm_client(provider: str | None = None) -> LLMClient:
    """Build the configured ``LLMClient``.

    A live provider is only constructed here; no credentials are read and no
    network call is made at wiring time. Missing credentials error out later,
    on the first extraction call.
    """
    name = (provider or config.llm_provider()).strip().lower()
    if name == "stub":
        return StubLLMClient(LLMExtractionOutput())
    if name == "gemini":
        return GeminiLLMClient()
    if name == "groq":
        return GroqLLMClient()
    raise ValueError(
        f"Unknown LLM_PROVIDER {name!r}. Choose one of "
        f"{', '.join(config.SUPPORTED_LLM_PROVIDERS)}."
    )