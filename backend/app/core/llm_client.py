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

SDK and network errors are translated into the typed exceptions in
app/core/errors.py, so callers never see google.genai or httpx exceptions.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.core.errors import (
    LLMAuthError,
    LLMConfigurationError,
    LLMEmptyResponseError,
    LLMError,
    LLMQuotaExceededError,
    LLMUnavailableError,
)

DEFAULT_TIMEOUT_SECONDS = 30.0


class LLMClient(ABC):
    """Interface every LLM backend (real or fake) implements."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send `prompt`, return the model's raw text response.

        Implementations raise subclasses of app.core.errors.LLMError."""
        raise NotImplementedError


def _parse_retry_after(details: Any) -> int | None:
    """
    Pull the suggested wait (seconds) out of a Google error payload's
    RetryInfo entry, e.g. {"@type": ".../google.rpc.RetryInfo",
    "retryDelay": "4951s"}. Returns None when absent or malformed, because
    a missing hint must never turn into a second error.
    """
    if not isinstance(details, dict):
        return None
    error = details.get("error", details)
    items = error.get("details") if isinstance(error, dict) else None
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        if not str(item.get("@type", "")).endswith("RetryInfo"):
            continue
        delay = item.get("retryDelay")
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return max(1, math.ceil(float(delay[:-1])))
            except ValueError:
                return None
    return None


def _translate_api_error(exc: genai_errors.APIError, model_name: str) -> LLMError:
    """Map a google-genai APIError to one of our typed exceptions."""
    code = exc.code
    message = (exc.message or "").strip()
    summary = f"Gemini API error {code} {exc.status or ''}: {message}".strip()

    if code == 429:
        return LLMQuotaExceededError(summary, retry_after_seconds=_parse_retry_after(exc.details))
    if code in (401, 403) or (code == 400 and "api key" in message.lower()):
        return LLMAuthError(summary)
    if code == 404:
        return LLMConfigurationError(
            f"Model {model_name!r} was not found or is not enabled for this API key. {summary}"
        )
    if code == 408 or code >= 500:
        return LLMUnavailableError(summary)
    return LLMError(summary)


class GeminiClient(LLMClient):
    """
    Real Gemini-backed client, using google-genai.

    model_name must be a model ID currently enabled for your API key (for
    example "gemini-2.5-flash"). Available models and free-tier limits
    change over time and differ by account, so verify what's enabled for
    your key at https://aistudio.google.com. The value in .env
    (GEMINI_MODEL) is what actually gets used.

    timeout_seconds bounds each request so a hung connection can't tie up a
    server worker indefinitely.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not api_key:
            raise LLMConfigurationError(
                "GEMINI_API_KEY is empty. Set it in backend/.env before making real calls."
            )
        self._client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )
        self._model_name = model_name

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
            )
        except genai_errors.APIError as exc:
            raise _translate_api_error(exc, self._model_name) from exc
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError(
                f"Request to Gemini timed out for model {self._model_name!r}."
            ) from exc
        except httpx.TransportError as exc:
            raise LLMUnavailableError(f"Could not reach the Gemini API: {exc}") from exc

        if not response.text:
            raise LLMEmptyResponseError(
                f"Gemini returned no text for model {self._model_name!r}. "
                "The response may have been blocked by safety filters."
            )
        return response.text


class FakeLLMClient(LLMClient):
    """
    Deterministic stand-in for tests and local development without an API
    key. Takes a callable (prompt -> str) or a fixed string, so tests can
    script exact responses, including malformed ones or raised errors for
    Step 4's retry logic.
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