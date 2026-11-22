"""
Tests for Step 2's app/core/llm_client.py.

GeminiClient tests mock google.genai entirely — no network calls, no API
key needed, no cost. They exist to catch wiring mistakes (wrong argument
names, wrong attribute read) against the real SDK shape, which a test
using only FakeLLMClient would never catch.

FakeLLMClient tests check its own contract, since Steps 2-7 all rely on
it behaving correctly in tests.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.core.llm_client import FakeLLMClient, GeminiClient


def test_gemini_client_rejects_empty_api_key():
    with pytest.raises(ValueError, match="GEMINI_API_KEY is empty"):
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
def test_gemini_client_raises_on_empty_response_text(mock_client_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text="")
    mock_client_cls.return_value = mock_client

    client = GeminiClient(api_key="fake-key", model_name="gemini-2.5-flash")
    with pytest.raises(ValueError, match="returned no text"):
        client.generate("question")


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