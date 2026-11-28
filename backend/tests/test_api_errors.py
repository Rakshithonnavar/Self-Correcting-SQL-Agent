"""
Tests that app/api/errors.py's exception handlers (registered in
app/main.py) actually produce the right HTTP response when exercised
through a real request — not just that the mapping function returns the
right exception type (that's covered in test_llm_client.py).

Uses the same StaticPool in-memory engine pattern as test_api_query.py —
see that file for why StaticPool is required with FastAPI's TestClient.
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from starlette.requests import Request

from app.api.errors import handle_agent_error
from app.core.errors import (
    LLMQuotaExceededError,
    LLMUnavailableError,
    QueryTimeoutError,
    UnsafeSQLError,
)
from app.core.llm_client import FakeLLMClient
from app.db.connection import get_engine
from app.dependencies import get_llm_client
from app.main import app


@pytest.fixture
def seeded_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE assets (id INTEGER PRIMARY KEY, name TEXT NOT NULL)"))
    return engine


@pytest.fixture
def empty_engine():
    """A reachable database with no tables at all — distinct from a
    database the app can't connect to."""
    return create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _client_with_llm(seeded_engine, llm):
    app.dependency_overrides[get_engine] = lambda: seeded_engine
    app.dependency_overrides[get_llm_client] = lambda: llm
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_quota_exceeded_returns_429_with_retry_after_header(seeded_engine):
    def responder(_prompt: str) -> str:
        raise LLMQuotaExceededError("daily quota used up", retry_after_seconds=42)

    client = _client_with_llm(seeded_engine, FakeLLMClient(responder=responder))
    response = client.post("/query", json={"question": "how many assets?"})

    assert response.status_code == 429
    assert response.headers["retry-after"] == "42"
    body = response.json()["detail"]
    assert body["code"] == "llm_quota_exceeded"
    assert body["retry_after_seconds"] == 42
    # The upstream message text must not leak verbatim to the client.
    assert "daily quota used up" not in response.text


def test_quota_exceeded_without_retry_hint_omits_header(seeded_engine):
    def responder(_prompt: str) -> str:
        raise LLMQuotaExceededError("quota used up")  # no retry_after_seconds

    client = _client_with_llm(seeded_engine, FakeLLMClient(responder=responder))
    response = client.post("/query", json={"question": "how many assets?"})

    assert response.status_code == 429
    assert "retry-after" not in response.headers
    assert response.json()["detail"]["code"] == "llm_quota_exceeded"


def test_llm_unavailable_returns_503(seeded_engine):
    def responder(_prompt: str) -> str:
        raise LLMUnavailableError("upstream is down")

    client = _client_with_llm(seeded_engine, FakeLLMClient(responder=responder))
    response = client.post("/query", json={"question": "anything"})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "llm_unavailable"


def test_empty_schema_returns_503_without_calling_the_llm(empty_engine):
    def responder(_prompt: str) -> str:
        raise AssertionError("the LLM must not be called when the schema is empty")

    client = _client_with_llm(empty_engine, FakeLLMClient(responder=responder))
    response = client.post("/query", json={"question": "how many assets?"})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "schema_empty"


def _make_request() -> Request:
    """A minimal Starlette Request, enough for handle_agent_error's logging
    (request.method, request.url.path) - no real HTTP connection needed."""
    return Request({"type": "http", "method": "POST", "path": "/query", "headers": []})


def test_unsafe_sql_error_fail_safe_mapping():
    """UnsafeSQLError is normally caught and corrected entirely inside
    app.core.correction_loop and never reaches this handler (see that
    module's docstring) - this tests the fail-safe mapping directly,
    rather than trying to force one past the loop through the full route."""
    response = asyncio.run(
        handle_agent_error(_make_request(), UnsafeSQLError("blocked for testing"))
    )
    assert response.status_code == 400
    body = json.loads(response.body)["detail"]
    assert body["code"] == "unsafe_sql_blocked"
    assert "blocked for testing" not in response.body.decode()


def test_query_timeout_error_fail_safe_mapping():
    """Same reasoning as test_unsafe_sql_error_fail_safe_mapping above:
    the correction loop normally retries this itself."""
    response = asyncio.run(
        handle_agent_error(_make_request(), QueryTimeoutError("took too long for testing"))
    )
    assert response.status_code == 504
    body = json.loads(response.body)["detail"]
    assert body["code"] == "query_timeout"
    assert "took too long for testing" not in response.body.decode()


def test_unexpected_exception_returns_generic_500_without_leaking_detail(seeded_engine):
    def responder(_prompt: str) -> str:
        raise RuntimeError("super secret internal stack trace detail")

    client = _client_with_llm(seeded_engine, FakeLLMClient(responder=responder))
    response = client.post("/query", json={"question": "anything"})

    assert response.status_code == 500
    body = response.json()["detail"]
    assert body["code"] == "internal_error"
    assert "super secret internal stack trace detail" not in response.text
