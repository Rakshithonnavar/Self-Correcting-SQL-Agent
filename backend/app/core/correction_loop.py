"""
Step 4: the self-correction loop this project is named for.

Wraps SQL generation + execution so that a failing or unsafe query is fed
back to the LLM — with the concrete error it produced — for another
attempt, instead of handing the first failure straight to the user.

Step 5 adds the actual guardrails this loop enforces on every attempt:
app.core.sql_safety.assert_safe_sql() parses the generated SQL, rejects
anything that isn't a single read-only SELECT over a table that exists in
the connected schema, and caps its result size — and
app.core.query_executor.execute_query_with_timeout() bounds how long the
database is allowed to spend on it. Both run inside the same try/except
as the database error that Step 4 already corrected, so a query that's
unsafe, out of bounds, or too slow gets the same "tell the model what
went wrong and let it try again" treatment as a plain syntax error.

Scope and safety boundaries, deliberately drawn here:
  - Three kinds of failure are corrected: the generated SQL executed but
    the database rejected it (QueryExecutionError); the generated SQL
    failed a guardrail — wrong statement type, an unknown table, or
    unparsable (UnsafeSQLError); or it ran too long and was cancelled
    (QueryTimeoutError). All three mean "the model wrote something that
    didn't work" — exactly what self-correction is for. A timeout is
    corrected rather than treated as a dead end because a narrower query
    (a filter, a smaller join) genuinely can fix it, unlike a provider
    outage.
  - LLM-provider failures (quota exceeded, auth, unavailable, empty
    response) and SQLGenerationError (nothing usable extracted from the
    raw response) are NOT retried here. They propagate immediately, same
    as before this loop existed. Retrying a quota-exhausted call would
    only spend more of a scarce daily quota on a call guaranteed to fail
    the same way.
  - Schema introspection and table selection happen once, by the caller,
    before this loop starts — not once per attempt. Only the SQL itself is
    regenerated on a retry; redoing the more expensive schema work on
    every attempt would multiply cost for no benefit. The table allowlist
    passed to assert_safe_sql is the schema's FULL table set, not the
    narrowed-down only_tables used to shrink the prompt — a query
    joining a table outside that narrowed subset is a prompting
    limitation, not a security violation, and must not be rejected as one.
  - Attempts are capped at max_attempts. If every attempt fails, this
    returns its last failure rather than raising — the caller decides what
    "giving up" looks like in the response (routes_query.py reports it in
    the `error` field, matching the pre-Step-4 contract for a bad query).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.engine import Engine

from app.core.errors import QueryTimeoutError, UnsafeSQLError
from app.core.llm_client import LLMClient
from app.core.query_executor import (
    QueryExecutionError,
    QueryResult,
    execute_query_with_timeout,
)
from app.core.schema_introspector import SchemaSnapshot
from app.core.sql_generator import generate_corrected_sql, generate_sql
from app.core.sql_safety import assert_safe_sql, dialect_for_engine

_CorrectableError = (UnsafeSQLError, QueryExecutionError, QueryTimeoutError)


@dataclass
class CorrectionAttempt:
    """One attempt's outcome, kept for transparency: visible in logs now,
    and the natural shape for Step 7's eval harness to report on later."""

    attempt_number: int
    sql: str
    error: str | None


@dataclass
class CorrectionResult:
    question: str
    final_sql: str
    result: QueryResult | None
    error: str | None
    truncated: bool = False
    attempts: list[CorrectionAttempt] = field(default_factory=list)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


def _failure_message(exc: Exception) -> str:
    """
    The text shown to the user (if every attempt fails) and fed back to
    the model as the next correction prompt's ERROR. For a
    QueryExecutionError this is the underlying database driver's own
    message, not the "Error executing SQL: ..." wrapper — matching the
    message format the API already returned before this loop existed.
    UnsafeSQLError and QueryTimeoutError's own messages are already
    client-appropriate as written.
    """
    if isinstance(exc, QueryExecutionError):
        return str(exc.original_error)
    return str(exc)


def generate_and_execute_with_correction(
    question: str,
    schema: SchemaSnapshot,
    llm: LLMClient,
    engine: Engine,
    only_tables: list[str] | None = None,
    max_attempts: int = 3,
    max_rows: int = 100,
    timeout_seconds: float = 10.0,
) -> CorrectionResult:
    """
    Generate SQL for `question`, validate and execute it, and — if it
    fails any guardrail, fails in the database, or times out — retry with
    the error fed back to the model, up to `max_attempts` total tries
    (the first generation counts as attempt 1).

    Raises whatever generate_sql()/generate_corrected_sql() raises for a
    non-correctable failure (an LLM-provider error, or SQLGenerationError).
    Never raises for a correctable failure (UnsafeSQLError,
    QueryExecutionError, QueryTimeoutError) — those are reported in the
    returned CorrectionResult.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    allowed_tables = set(schema.table_names())
    dialect = dialect_for_engine(engine)

    sql = generate_sql(question, schema, llm, only_tables=only_tables)
    attempts: list[CorrectionAttempt] = []

    for attempt_number in range(1, max_attempts + 1):
        # What gets recorded/retried if this attempt fails: the model's
        # own SQL until assert_safe_sql produces a validated, row-capped
        # version of it — reassigned below the instant that succeeds, so
        # a validation failure still reports the SQL that was actually
        # rejected, and an execution/timeout failure reports what was
        # actually sent to the database.
        attempted_sql = sql
        try:
            safe_sql = assert_safe_sql(sql, allowed_tables, max_rows, dialect=dialect)
            attempted_sql = safe_sql
            result = execute_query_with_timeout(safe_sql, engine, timeout_seconds)
        except _CorrectableError as exc:
            error_message = _failure_message(exc)
            attempts.append(CorrectionAttempt(attempt_number, attempted_sql, error_message))

            if attempt_number >= max_attempts:
                return CorrectionResult(
                    question=question,
                    final_sql=attempted_sql,
                    result=None,
                    error=error_message,
                    attempts=attempts,
                )

            sql = generate_corrected_sql(
                question,
                schema,
                llm,
                failed_sql=attempted_sql,
                error_message=error_message,
                only_tables=only_tables,
            )
        else:
            truncated = result.row_count > max_rows
            if truncated:
                result = QueryResult(
                    columns=result.columns,
                    rows=result.rows[:max_rows],
                    row_count=max_rows,
                )
            attempts.append(CorrectionAttempt(attempt_number, safe_sql, None))
            return CorrectionResult(
                question=question,
                final_sql=safe_sql,
                result=result,
                error=None,
                truncated=truncated,
                attempts=attempts,
            )

    raise AssertionError("unreachable: the loop above always returns")  # pragma: no cover