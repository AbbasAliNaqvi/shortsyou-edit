"""Runtime configuration for the edit worker.

Secrets are deliberately read from the environment. Do not commit API keys to
this module or to source control.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_groq_keys() -> list[str]:
    """Return all non-empty Groq keys from environment, deduplicated."""
    keys = []
    for var in ("GROQ_KEY_1", "GROQ_KEY_2", "GROQ_API_KEY"):
        val = (os.getenv(var) or "").strip()
        if val and val not in keys:
            keys.append(val)
    return keys


def get_gemini_keys() -> list[str]:
    """Return all non-empty Gemini keys from environment, deduplicated."""
    keys = []
    for var in ("GEMINI_KEY_1", "GEMINI_KEY_2", "GEMINI_API_KEY"):
        val = (os.getenv(var) or "").strip()
        if val and val not in keys:
            keys.append(val)
    return keys


def groq_api_key() -> str | None:
    """Return the primary Groq key while retaining legacy worker support."""
    keys = get_groq_keys()
    return keys[0] if keys else None


GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "6.0"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
