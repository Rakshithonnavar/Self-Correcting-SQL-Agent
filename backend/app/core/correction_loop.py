"""
Step 4: the self-correction loop this project is named for.

Wraps SQL generation + execution so that a failing or unsafe query is fed
back to the LLM — with the concrete error it produced — for another
attempt, instead of handing the first failure straight to the user.

Scope and safety boundaries, deliberately drawn here:
  - Only two kinds of failure are corrected: the generated SQL executed but
    the database rejected it (QueryExecutionError), or the generated SQL
    was not a read-only SELECT/WITH statement (UnsafeSQLError, checked by
    app.core.sql_safety before every attempt). Both mean "the model wrote
    something wrong" — exactly what self-correction is for.
  - LLM-provider failures (quota exceeded, auth, unavailable, empty
    response) and SQLGenerationError (nothing usable extracted from the
    raw response) are NOT retried here. They propagate immediately, same
    as before this loop existed. Retrying a quota-exhausted call would
    only spend more of a scarce daily quota on a call guaranteed to fail
    the same way.
  - Schema introspection and table selection happen once, by the caller,
    before this loop starts — not once per attempt. Only the SQL itself is
    regenerated on a retry; redoing the more expensive schema work on
    every attempt would multiply cost for no benefit.
  - Attempts are capped at max_attempts. If every attempt fails, this
    returns its last failure rather than raising — the caller decides what
    "giving up" looks like in the response (routes_query.py reports it in
    the `error` field, matching the pre-Step-4 contract for a bad query).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.engine import Engine

from app.core.errors import UnsafeSQLError
from app.core.llm_client import LLMClient
from app.core.query_executor import QueryExecutionError, QueryResult, execute_query
from app.core.schema_introspector import SchemaSnapshot
from app.core.sql_generator import generate_corrected_sql, generate_sql
from app.core.sql_safety import ensure_select_only


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
    attempts: list[CorrectionAttempt] = field(default_factory=list)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


def _failure_message(exc: UnsafeSQLError | QueryExecutionError) -> str:
    """
    The text shown to the user (if every attempt fails) and fed back to
    the model as the next correction prompt's ERROR. For a QueryExecutionError
    this is the underlying database driver's own message, not the
    "Error executing SQL: ..." wrapper — matching the message format the
    API already returned before this loop existed.
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
) -> CorrectionResult:
    """
    Generate SQL for `question`, execute it, and — if it fails or is
    unsafe — retry with the error fed back to the model, up to
    `max_attempts` total tries (the first generation counts as attempt 1).

    Raises whatever generate_sql()/generate_corrected_sql() raises for a
    non-correctable failure (an LLM-provider error, or SQLGenerationError).
    Never raises for a correctable failure (QueryExecutionError,
    UnsafeSQLError) — those are reported in the returned CorrectionResult.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    sql = generate_sql(question, schema, llm, only_tables=only_tables)
    attempts: list[CorrectionAttempt] = []

    for attempt_number in range(1, max_attempts + 1):
        try:
            ensure_select_only(sql)
            result = execute_query(sql, engine)
        except (UnsafeSQLError, QueryExecutionError) as exc:
            error_message = _failure_message(exc)
            attempts.append(CorrectionAttempt(attempt_number, sql, error_message))

            if attempt_number >= max_attempts:
                return CorrectionResult(
                    question=question,
                    final_sql=sql,
                    result=None,
                    error=error_message,
                    attempts=attempts,
                )

            sql = generate_corrected_sql(
                question,
                schema,
                llm,
                failed_sql=sql,
                error_message=error_message,
                only_tables=only_tables,
            )
        else:
            attempts.append(CorrectionAttempt(attempt_number, sql, None))
            return CorrectionResult(
                question=question,
                final_sql=sql,
                result=result,
                error=None,
                attempts=attempts,
            )

    raise AssertionError("unreachable: the loop above always returns")  # pragma: no cover
