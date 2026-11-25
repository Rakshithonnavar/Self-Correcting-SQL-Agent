"""
Tests for Step 2's POST /query endpoint (app/api/routes_query.py).

Overrides both FastAPI dependencies (get_engine, get_llm_client) so this
test suite never touches a real Gemini API and never touches dev.db —
it runs against an isolated in-memory SQLite engine seeded per test.

poolclass=StaticPool matters here: FastAPI's TestClient runs the request
(including dependency resolution) in a worker thread, not the thread that
seeded the fixture. A plain sqlite:///:memory: engine hands out a FRESH,
empty in-memory database to each new connection/thread, so the seeded row
silently disappears. StaticPool forces every checkout to reuse the same
single connection, so the seeded data is actually visible to the request.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

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
        conn.execute(text("INSERT INTO assets (id, name) VALUES (1, 'payments-api')"))
    return engine


@pytest.fixture
def client(seeded_engine):
    fake_llm = FakeLLMClient(response="SELECT * FROM assets")
    app.dependency_overrides[get_engine] = lambda: seeded_engine
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    yield TestClient(app), fake_llm
    app.dependency_overrides.clear()


def test_health_endpoint():
    test_client = TestClient(app)
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint_returns_rows(client):
    test_client, _ = client
    response = test_client.post("/query", json={"question": "Show me all assets"})
    assert response.status_code == 200
    body = response.json()
    assert body["sql"] == "SELECT * FROM assets"
    assert body["row_count"] == 1
    assert body["rows"] == [{"id": 1, "name": "payments-api"}]
    assert body["error"] is None


def test_query_endpoint_sends_question_to_llm(client):
    test_client, fake_llm = client
    test_client.post("/query", json={"question": "Show me all assets"})
    assert len(fake_llm.calls) == 1
    assert "Show me all assets" in fake_llm.calls[0]


def test_query_endpoint_returns_error_on_bad_sql(client):
    test_client, fake_llm = client
    fake_llm._response = "SELECT * FROM table_that_does_not_exist"
    response = test_client.post("/query", json={"question": "anything"})
    assert response.status_code == 200  # Step 2: no retry yet, so this surfaces as a field
    body = response.json()
    assert body["error"] is not None
    assert body["rows"] == []

def test_query_endpoint_self_corrects_a_bad_table_name(client):
    test_client, fake_llm = client
    responses = iter(["SELECT * FROM asets", "SELECT * FROM assets"])
    fake_llm._responder = lambda _prompt: next(responses)
    fake_llm._response = None

    response = test_client.post("/query", json={"question": "show me all assets"})
    assert response.status_code == 200
    body = response.json()
    assert body["error"] is None
    assert body["row_count"] == 1
    assert body["sql"] == "SELECT * FROM assets"
    assert body["attempts"] == 2


def test_query_endpoint_reports_attempts_as_one_on_first_try_success(client):
    test_client, _ = client
    response = test_client.post("/query", json={"question": "show me all assets"})
    assert response.json()["attempts"] == 1


def test_query_endpoint_blocks_an_unsafe_generated_statement(client, seeded_engine):
    test_client, fake_llm = client
    fake_llm._response = "DELETE FROM assets"

    response = test_client.post("/query", json={"question": "anything"})
    assert response.status_code == 200  # exhausted corrections -> reported in body, not raised
    body = response.json()
    assert body["error"] is not None
    assert "not a read-only SELECT" in body["error"]

    # Confirms the unsafe statement never reached the database, by
    # checking the row directly rather than through the same fake LLM
    # (which would just produce the same blocked statement again).
    with seeded_engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM assets")).scalar() == 1

def test_query_endpoint_requires_question_field(client):
    # Uses the overridden `client` fixture (not a bare TestClient) so this
    # test checks ONLY request validation, without also exercising the
    # real GeminiClient dependency (which would fail first on an empty key).
    test_client, _ = client
    response = test_client.post("/query", json={})
    assert response.status_code == 422
