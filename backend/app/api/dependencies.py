"""
FastAPI dependency providers.

Kept separate from app/config.py and app/core/llm_client.py so routes can
depend on `get_llm_client` and tests can override it with a FakeLLMClient
via app.dependency_overrides, without touching real settings or the real
Gemini SDK.
"""

from functools import lru_cache

from app.config import get_settings
from app.core.llm_client import GeminiClient, LLMClient


@lru_cache
def get_llm_client() -> LLMClient:
    settings = get_settings()
    return GeminiClient(api_key=settings.gemini_api_key, model_name=settings.gemini_model)