"""
Step 2: thin wrapper around the Gemini API, via the google-genai SDK
(the current, actively-maintained client — not the older
google-generativeai package, which Google is phasing out).

This is the ONLY module that imports google.genai. Everything else
(sql_generator, self-correction in Step 4) calls LLMClient, not Gemini
directly. That makes two things possible later without touching the rest
of the codebase:
  - swapping providers (e.g. adding an OpenAI fallback)
  - testing without a real API key, via FakeLLMClient below
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable


class LLMClient(ABC):
    """Interface every LLM backend (real or fake) implements."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send `prompt`, return the model's raw text response."""
        raise NotImplementedError


class GeminiClient(LLMClient):
    """
    Real Gemini-backed client, using google-genai.

    model_name must be a model ID currently enabled for your API key (for
    example "gemini-2.5-flash"). Available models and free-tier limits
    change over time and differ by account, so verify what's enabled for
    your key at https://aistudio.google.com before relying on a default —
    the value in .env (GEMINI_MODEL) is what actually gets used, this
    class's default is only a fallback if that's left unset.
    """

    def __init__(self, api_key: str, model_name: str) -> None:
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is empty. Set it in backend/.env before making real calls."
            )
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
        )
        if not response.text:
            raise ValueError(
                f"Gemini returned no text for model {self._model_name!r}. "
                "This can happen if the response was blocked by safety filters "
                "or the model name is invalid/unavailable for this API key."
            )
        return response.text


class FakeLLMClient(LLMClient):
    """
    Deterministic stand-in for tests and local development without an API
    key. Takes a callable (prompt -> str) or a fixed string, so tests can
    script exact responses, including malformed ones for Step 4's retry
    logic.
    """

    def __init__(
        self,
        response: str | None = None,
        responder: Callable[[str], str] | None = None,
    ) -> None:
        if response is None and responder is None:
            raise ValueError("FakeLLMClient needs either `response` or `responder`.")
        self._response = response
        self._responder = responder
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self._responder is not None:
            return self._responder(prompt)
        assert self._response is not None  # guaranteed by __init__'s check
        return self._response