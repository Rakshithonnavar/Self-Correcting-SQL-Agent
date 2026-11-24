"""
Tests for Step 2's app/core/llm_client.py.

GeminiClient tests mock google.genai entirely — no network calls, no API
key needed, no cost. They exist to catch wiring mistakes (wrong argument
names, wrong attribute read) against the real SDK shape, which a test
using only FakeLLMClient would never catch.

Error-translation tests build REAL google.genai.errors.APIError instances
(via the SDK's own raise_error classmethod, with a fake HTTP response),
rather than hand-rolling exceptions with guessed attributes — so these
tests fail if the SDK's error shape ever changes, instead of passing
against a mock that no longer matches reality.

FakeLLMClient tests check its own contract, since Steps 2-7 all rely on
it behaving correctly in tests.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest
from google.genai import errors as genai_errors

from app.core.errors import (
    LLMAuthError,
    LLMConfigurationError,
    LLMEmptyResponseError,
    LLMError,
    LLMQuotaExceededError,
    LLMUnavailableError,
)
from app.core.llm_client import FakeLLMClient, GeminiClient, _translate_api_error


def _make_api_error(status_code: int, payload: dict) -> genai_errors.APIError:
    """Build a real APIError the way the SDK itself does, using a minimal
    fake httpx.Response so we don't guess at APIError's internal shape."""
    request = httpx.Request("POST", "https://example.invalid/v1/generate")
    response = httpx.Response(status_code, json=payload, request=request)
    try:
        genai_errors.APIError.raise_error(status_code, payload, response)
    except genai_errors.APIError as exc:
        return exc
    raise AssertionError("raise_error did not raise")  # pragma: no cover


def _quota_payload(retry_delay: str = "60s") -> dict:
    return {
        "error": {
            "code": 429,
            "message": "You exceeded your current quota.",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay},
            ],
        }
    }


# --- GeminiClient construction and happy path -------------------------------


def test_gemini_client_rejects_empty_api_key():
    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY is empty"):
        GeminiClient(api_key="", model_name="gemini-2.5-flash")


@patch("google.genai.Client")
def test_gemini_client_calls_generate_content_with_model_and_prompt(mock_client_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text="SELECT 1")
    mock_client_cls.return_value = mock_client

    client = GeminiClient(api_key="fake-key", model_name="gemini-2.5-flash")
    result = client.generate("How many assets?")

    assert result == "SELECT 1"
    mock_client.models.generate_content.assert_called_once_with(
        model="gemini-2.5-flash",
        contents="How many assets?",
    )


@patch("google.genai.Client")
def test_gemini_client_raises_llm_empty_response_error_on_blank_text(mock_client_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text="")
    mock_client_cls.return_value = mock_client

    client = GeminiClient(api_key="fake-key", model_name="gemini-2.5-flash")
    with pytest.raises(LLMEmptyResponseError, match="returned no text"):
        client.generate("question")


# --- Error translation: HTTP/SDK errors -> our typed exceptions -------------


def test_429_translates_to_quota_exceeded_with_retry_after():
    exc = _make_api_error(429, _quota_payload(retry_delay="90s"))
    result = _translate_api_error(exc, model_name="gemini-2.5-flash")
    assert isinstance(result, LLMQuotaExceededError)
    assert result.retry_after_seconds == 90


def test_429_without_retry_info_still_translates_with_no_retry_hint():
    payload = {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}
    exc = _make_api_error(429, payload)
    result = _translate_api_error(exc, model_name="gemini-2.5-flash")
    assert isinstance(result, LLMQuotaExceededError)
    assert result.retry_after_seconds is None


@pytest.mark.parametrize("status_code", [401, 403])
def test_401_403_translate_to_auth_error(status_code):
    payload = {"error": {"code": status_code, "message": "permission denied", "status": "X"}}
    exc = _make_api_error(status_code, payload)
    result = _translate_api_error(exc, model_name="gemini-2.5-flash")
    assert isinstance(result, LLMAuthError)


def test_404_translates_to_configuration_error_naming_the_model():
    payload = {"error": {"code": 404, "message": "model not found", "status": "NOT_FOUND"}}
    exc = _make_api_error(404, payload)
    result = _translate_api_error(exc, model_name="gemini-made-up-model")
    assert isinstance(result, LLMConfigurationError)
    assert "gemini-made-up-model" in str(result)


@pytest.mark.parametrize("status_code", [500, 503])
def test_5xx_translates_to_unavailable_error(status_code):
    payload = {"error": {"code": status_code, "message": "server error", "status": "X"}}
    exc = _make_api_error(status_code, payload)
    result = _translate_api_error(exc, model_name="gemini-2.5-flash")
    assert isinstance(result, LLMUnavailableError)


def test_unmapped_status_falls_back_to_generic_llm_error():
    payload = {"error": {"code": 418, "message": "teapot", "status": "X"}}
    exc = _make_api_error(418, payload)
    result = _translate_api_error(exc, model_name="gemini-2.5-flash")
    assert type(result) is LLMError


@patch("google.genai.Client")
def test_gemini_client_translates_api_error_raised_by_sdk(mock_client_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = _make_api_error(429, _quota_payload())
    mock_client_cls.return_value = mock_client

    client = GeminiClient(api_key="fake-key", model_name="gemini-2.5-flash")
    with pytest.raises(LLMQuotaExceededError):
        client.generate("question")


@patch("google.genai.Client")
def test_gemini_client_translates_timeout_to_unavailable(mock_client_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = httpx.TimeoutException("timed out")
    mock_client_cls.return_value = mock_client

    client = GeminiClient(api_key="fake-key", model_name="gemini-2.5-flash")
    with pytest.raises(LLMUnavailableError, match="timed out"):
        client.generate("question")


@patch("google.genai.Client")
def test_gemini_client_translates_transport_error_to_unavailable(mock_client_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = httpx.ConnectError("connection refused")
    mock_client_cls.return_value = mock_client

    client = GeminiClient(api_key="fake-key", model_name="gemini-2.5-flash")
    with pytest.raises(LLMUnavailableError, match="Could not reach"):
        client.generate("question")


# --- FakeLLMClient -----------------------------------------------------------


def test_fake_llm_client_requires_response_or_responder():
    with pytest.raises(ValueError, match="needs either"):
        FakeLLMClient()


def test_fake_llm_client_returns_fixed_response():
    llm = FakeLLMClient(response="SELECT * FROM assets")
    assert llm.generate("any prompt") == "SELECT * FROM assets"


def test_fake_llm_client_records_calls():
    llm = FakeLLMClient(response="SELECT 1")
    llm.generate("first")
    llm.generate("second")
    assert llm.calls == ["first", "second"]


def test_fake_llm_client_uses_responder_when_given():
    llm = FakeLLMClient(responder=lambda prompt: f"echo: {prompt}")
    assert llm.generate("hello") == "echo: hello"


def test_fake_llm_client_responder_can_raise_to_simulate_failures():
    def responder(_prompt: str) -> str:
        raise LLMQuotaExceededError("simulated quota error", retry_after_seconds=30)

    llm = FakeLLMClient(responder=responder)
    with pytest.raises(LLMQuotaExceededError):
        llm.generate("question")