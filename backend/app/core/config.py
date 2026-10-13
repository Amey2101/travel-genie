"""Environment-based configuration.

Ported and adapted from the teammate implementation's ``settings.py``.

Keys are read at call time so tests can override them via the environment.
Secret values are never logged or returned by the API.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[3]
# Canonical local configuration file: <repo-root>/.env. This is the single
# source of truth; backend/.env is deliberately not read.
_ENV_PATH = _REPO_ROOT / ".env"

# override=False: explicitly exported shell variables always win over the file.
# Real credentials in the environment are therefore never clobbered by stale or
# blank placeholder values left in .env.
if _ENV_PATH.is_file():
    load_dotenv(_ENV_PATH, override=False)


def env_file_found() -> bool:
    return _ENV_PATH.is_file()


def log_level() -> str:
    return os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO"


# ---------------------------------------------------------------------------
# LLM providers (Gemini primary, Groq fallback). Config only — clients are
# implemented in a later milestone.
# ---------------------------------------------------------------------------

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

# Supported structured-extraction providers. Selection is explicit and never
# falls back automatically; `stub` is the credential-free default.
SUPPORTED_LLM_PROVIDERS = ("stub", "gemini", "groq")


def llm_provider() -> str:
    """Selected LLM provider name (lower-cased). Defaults to ``stub``."""
    return os.getenv("LLM_PROVIDER", "stub").strip().lower() or "stub"


def gemini_key() -> str:
    return os.getenv("GEMINI_API_KEY", "").strip()


def gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL


def groq_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()


def groq_model() -> str:
    return os.getenv("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL


def serpapi_key() -> str:
    return os.getenv("SERPAPI_API_KEY", "").strip()


def _positive_number(name: str, default: float, minimum: float, maximum: float) -> float:
    """Read a bounded positive numeric setting, falling back safely on bad input."""
    try:
        value = float(os.getenv(name, str(default)).strip())
    except (TypeError, ValueError):
        return default
    return value if minimum <= value <= maximum else default


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except (TypeError, ValueError):
        return default
    return value if minimum <= value <= maximum else default


def gemini_timeout_seconds() -> float:
    """HTTP timeout for Gemini, configurable from .env (10-120 seconds)."""
    return _positive_number("GEMINI_TIMEOUT_SECONDS", 35.0, 10.0, 120.0)


def gemini_max_output_tokens() -> int:
    """Output token cap for Gemini, configurable from .env (1024-8192 tokens)."""
    return _bounded_int("GEMINI_MAX_OUTPUT_TOKENS", 8192, 1024, 8192)


def groq_timeout_seconds() -> float:
    """HTTP timeout for Groq, configurable from .env (10-120 seconds)."""
    return _positive_number("GROQ_TIMEOUT_SECONDS", 35.0, 10.0, 120.0)


def groq_max_output_tokens() -> int:
    """Output token cap for Groq, configurable from .env (1024-16384 tokens)."""
    return _bounded_int("GROQ_MAX_OUTPUT_TOKENS", 8192, 1024, 16384)


def serpapi_timeout_seconds() -> float:
    """Network timeout for SerpApi discovery, configurable from .env (5-60 seconds)."""
    return _positive_number("SERPAPI_TIMEOUT_SECONDS", 15.0, 5.0, 60.0)
