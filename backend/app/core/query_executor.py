"""
Step 2: executes a SQL string against the live database and returns rows.

Step 5 adds execute_query_with_timeout(), which enforces a wall-clock
limit on top of the same execution path, and cancels the query at the
database driver level (not just abandoning it from Python) when it's
exceeded. execute_query() itself is unchanged and still used directly by
the timeout wrapper, so there's exactly one place that turns a SQLAlchemy
result into a QueryResult.

This module does NOT enforce SELECT-only, a table allowlist, or a row
cap - that's app.core.sql_safety, which runs before a SQL string ever
reaches this module. By the time execute_query[_with_timeout] sees a
query, Step 5's correction loop has already validated and row-limited it;
this module's job is only to run it and shape the result.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from app.core.errors import QueryTimeoutError


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


def _fetch_result(conn: Connection, sql: str) -> QueryResult:
    """Shared by execute_query() and execute_query_with_timeout()'s worker,
    so there's one place that turns a SQLAlchemy result into a QueryResult."""
    result = conn.execute(text(sql))
    columns = list(result.keys())
    rows = [dict(zip(columns, row)) for row in result.fetchall()]
    return QueryResult(columns=columns, rows=rows, row_count=len(rows))


def execute_query(sql: str, engine: Engine) -> QueryResult:
    """
    Run `sql` against `engine` and return the result as plain Python
    data (list of dict rows), not SQLAlchemy Row objects, so callers
    (the API layer, the eval harness) don't need to import SQLAlchemy.
    """
    try:
        with engine.connect() as conn:
            return _fetch_result(conn, sql)
    except SQLAlchemyError as exc:
        raise QueryExecutionError(sql, exc) from exc


def _cancel_raw_connection(raw_conn: Any) -> None:
    """
    Best-effort query cancellation, using whichever primitive the DBAPI
    driver exposes: psycopg2 connections have .cancel() (sends a cancel
    request to Postgres on a separate link); sqlite3 connections have
    .interrupt() (flags the connection so its next VM step raises).
    Never raises - if neither exists, or calling it fails, the timeout
    error below is still returned; the worker thread then finishes
    whenever the database itself gives up, not before.
    """
    for method_name in ("cancel", "interrupt"):
        method = getattr(raw_conn, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:  # noqa: BLE001 - cancellation is best-effort, never fatal
                pass
            return


def execute_query_with_timeout(sql: str, engine: Engine, timeout_seconds: float) -> QueryResult:
    """
    Like execute_query(), but raises QueryTimeoutError instead of letting
    a slow query run indefinitely. The query runs in a single-use worker
    thread; on timeout, the underlying DBAPI connection is cancelled via
    _cancel_raw_connection() so the database itself stops working on it,
    not just the Python code waiting on it.

    Known limitation, stated plainly rather than hidden: cancellation
    needs a reference to the worker's raw connection, captured the instant
    it connects. If the timeout fires in the narrow window before that
    connection is even open (vanishingly unlikely next to how long an
    actual slow query runs, but possible under a very small
    timeout_seconds), this can't cancel it and the worker thread keeps
    running in the background until the database finishes or errors on
    its own - this function still returns QueryTimeoutError either way, it
    just can't guarantee the work actually stopped.

    A fresh single-worker thread pool is created per call rather than
    shared across requests. That's a few microseconds of overhead, which
    is negligible next to the database round trip it's wrapping, and it
    avoids the lifecycle complexity of a pool shared across concurrent
    requests with independent timeouts.

    SQLite in-memory engines, specifically: this function always opens a
    NEW connection inside its worker thread, deliberately (so it can grab
    that connection's raw handle for cancellation). A plain
    "sqlite:///:memory:" engine hands a brand-new, empty database to each
    new connection in each new thread, so without poolclass=StaticPool on
    that engine, the worker thread would be querying a different database
    than whatever the caller set up. This is specific to :memory: engines
    (a real file, or Neon/Postgres in production, is the same database no
    matter which thread connects to it) — but it's exactly the kind of
    thing that looks like a mystery "table not found" if missed, so it's
    worth knowing up front rather than debugging it from a stack trace.
    """
    raw_connection_box: list[Any] = []

    def _run() -> QueryResult:
        with engine.connect() as conn:
            raw_connection_box.append(conn.connection.dbapi_connection)
            return _fetch_result(conn, sql)

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_run)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            if raw_connection_box:
                _cancel_raw_connection(raw_connection_box[0])
            raise QueryTimeoutError(
                f"Query did not complete within {timeout_seconds:g}s and was cancelled."
            ) from None
        except SQLAlchemyError as exc:
            raise QueryExecutionError(sql, exc) from exc
    finally:
        # wait=False: don't block the request on a worker thread that a
        # just-issued cancellation is still unwinding.
        executor.shutdown(wait=False)
