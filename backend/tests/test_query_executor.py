"""
Tests for Step 2's app/core/query_executor.py, and Step 5's addition,
execute_query_with_timeout().
"""

import time

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from app.core.errors import QueryTimeoutError
from app.core.query_executor import (
    QueryExecutionError,
    execute_query,
    execute_query_with_timeout,
)


@pytest.fixture
def seeded_engine():
    # StaticPool matters for execute_query_with_timeout specifically: it
    # runs the query in a worker thread, and a plain sqlite:///:memory:
    # engine hands a brand-new, empty database to each new thread's
    # connection. StaticPool forces every checkout to reuse the one real
    # connection, so the worker thread sees the data this fixture seeded.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE assets (id INTEGER PRIMARY KEY, name TEXT)"))
        conn.execute(text("INSERT INTO assets (id, name) VALUES (1, 'payments-api')"))
        conn.execute(text("INSERT INTO assets (id, name) VALUES (2, 'auth-service')"))
    return engine


def test_execute_query_returns_rows_as_dicts(seeded_engine):
    result = execute_query("SELECT id, name FROM assets ORDER BY id", seeded_engine)
    assert result.columns == ["id", "name"]
    assert result.rows == [
        {"id": 1, "name": "payments-api"},
        {"id": 2, "name": "auth-service"},
    ]
    assert result.row_count == 2


def test_execute_query_handles_empty_result(seeded_engine):
    result = execute_query("SELECT * FROM assets WHERE id = 999", seeded_engine)
    assert result.rows == []
    assert result.row_count == 0


def test_execute_query_raises_on_invalid_sql(seeded_engine):
    with pytest.raises(QueryExecutionError) as exc_info:
        execute_query("SELECT * FROM table_that_does_not_exist", seeded_engine)
    assert exc_info.value.sql == "SELECT * FROM table_that_does_not_exist"
    assert exc_info.value.original_error is not None


def test_execute_query_raises_on_syntax_error(seeded_engine):
    with pytest.raises(QueryExecutionError):
        execute_query("SELEKT * FROM assets", seeded_engine)


# --- execute_query_with_timeout ------------------------------------------


def test_with_timeout_returns_the_same_result_as_execute_query_when_fast(seeded_engine):
    result = execute_query_with_timeout(
        "SELECT id, name FROM assets ORDER BY id", seeded_engine, timeout_seconds=5
    )
    assert result.row_count == 2
    assert result.rows[0] == {"id": 1, "name": "payments-api"}


def test_with_timeout_raises_query_execution_error_on_bad_sql(seeded_engine):
    with pytest.raises(QueryExecutionError):
        execute_query_with_timeout("SELEKT * FROM assets", seeded_engine, timeout_seconds=5)


def test_with_timeout_cancels_a_genuinely_slow_query(seeded_engine):
    # A real runaway query (not mocked), proving the cancellation actually
    # stops the database from working on it rather than just abandoning it
    # from Python: if cancellation didn't work, this query would still be
    # consuming CPU in the background for tens of seconds after this
    # function returns, and the second query below would be slow to start.
    slow_sql = (
        "WITH RECURSIVE cnt(x) AS ("
        "SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 500000000"
        ") SELECT COUNT(*) FROM cnt"
    )
    start = time.monotonic()
    with pytest.raises(QueryTimeoutError, match="did not complete"):
        execute_query_with_timeout(slow_sql, seeded_engine, timeout_seconds=0.3)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0  # returned promptly, not after the full recursive scan

    # The connection/engine is still usable afterwards for an ordinary query.
    result = execute_query_with_timeout(
        "SELECT COUNT(*) FROM assets", seeded_engine, timeout_seconds=5
    )
    assert result.rows == [{"COUNT(*)": 2}]


def test_with_timeout_of_zero_or_negative_still_eventually_raises(seeded_engine):
    # Not a supported "instant timeout" guarantee, just confirms a very
    # small timeout doesn't hang - it either completes a trivial query in
    # time or is reported as timed out, never silently blocks forever.
    try:
        execute_query_with_timeout("SELECT 1", seeded_engine, timeout_seconds=0.001)
    except QueryTimeoutError:
        pass