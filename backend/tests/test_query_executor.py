"""
Tests for Step 2's app/core/query_executor.py.
"""

import pytest
from sqlalchemy import create_engine, text

from app.core.query_executor import QueryExecutionError, execute_query


@pytest.fixture
def seeded_engine():
    engine = create_engine("sqlite:///:memory:")
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
