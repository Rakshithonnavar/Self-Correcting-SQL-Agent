"""
Step 2: executes a SQL string against the live database and returns rows.

Not in the original step list, added because Steps 2, 4, 5, and 6 all
need the same "run this SQL, get rows or an error" primitive. Keeping it
in one place means Step 5's guardrails (timeout, row cap) and Step 4's
self-correction (catching the error) both wrap this single function.

This module does NOT enforce SELECT-only or any other guardrail — that
is Step 5. As of Step 2, it will happily run whatever SQL it's given.
Do not point this at a database with real data until Step 5 lands.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[dict]
    row_count: int


class QueryExecutionError(Exception):
    """Wraps the underlying DB error with the SQL that caused it, so
    Step 4's self-correction loop has both pieces it needs to retry."""

    def __init__(self, sql: str, original_error: Exception) -> None:
        self.sql = sql
        self.original_error = original_error
        super().__init__(f"Error executing SQL: {original_error}")


def execute_query(sql: str, engine: Engine) -> QueryResult:
    """
    Run `sql` against `engine` and return the result as plain Python
    data (list of dict rows), not SQLAlchemy Row objects, so callers
    (the API layer, the eval harness) don't need to import SQLAlchemy.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchall()]
            return QueryResult(columns=columns, rows=rows, row_count=len(rows))
    except SQLAlchemyError as exc:
        raise QueryExecutionError(sql, exc) from exc