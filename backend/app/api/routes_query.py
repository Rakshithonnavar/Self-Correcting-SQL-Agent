"""
POST /query — the text-to-SQL endpoint.

Step 2 gave this a naive, no-retry version: question in, SQL out, SQL
executed, result (or error) back.

Step 3 added smart schema selection (app/core/schema_selector.py), so the
LLM sees only the tables relevant to the question, plus anything directly
joined to them, instead of the full schema on every call — and a guard
against an empty schema, raised before the LLM is ever called.

Step 4 replaces the "run it once" execution with
app.core.correction_loop.generate_and_execute_with_correction(): a failing
or unsafe query is fed back to the model, with its exact error, for up to
settings.max_correction_attempts tries. Schema introspection and table
selection still happen once per request here, not once per attempt — only
the correction loop's internal retries regenerate the SQL itself.

Step 5 passes settings.max_rows and settings.query_timeout_seconds into
that same call: the correction loop now validates the generated SQL
(single read-only SELECT, only known tables), caps how many rows come
back, and cancels a query that runs too long — all enforced on every
attempt, not just the first. The response's `truncated` field tells the
caller when more rows existed than settings.max_rows allows back.

Error handling: get_schema_snapshot() and the correction loop can raise
EmptySchemaError, LLMError subclasses (quota exceeded, auth failure,
provider unavailable, empty response), or SQLGenerationError. None of
those are caught here — they propagate to the handlers registered in
app/api/errors.py, which turn each into the right HTTP status and a
client-safe JSON body. Catching them per-route would mean duplicating
that mapping in every endpoint that calls the LLM.

A query that still fails after every correction attempt is NOT raised as
an HTTP error: it's reported in the response body's `error` field with a
200 status, exactly as a single failing query was reported before Step 4
existed. The `attempts` field tells the caller how many tries it took (1
means it worked first try; anything higher means self-correction kicked
in).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.engine import Engine

from app.config import Settings, get_settings
from app.core.correction_loop import generate_and_execute_with_correction
from app.core.errors import EmptySchemaError
from app.core.llm_client import LLMClient
from app.core.schema_introspector import get_schema_snapshot
from app.core.schema_selector import select_relevant_tables
from app.db.connection import get_engine
from app.dependencies import get_llm_client

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    sql: str
    columns: list[str] = []
    rows: list[dict] = []
    row_count: int = 0
    error: str | None = None
    attempts: int = 1
    truncated: bool = False


@router.post("/query", response_model=QueryResponse)
def run_query(
    request: QueryRequest,
    engine: Engine = Depends(get_engine),
    llm: LLMClient = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    schema = get_schema_snapshot(engine)
    if not schema.table_names():
        raise EmptySchemaError(
            "The connected database has no tables. Seed or migrate it before querying."
        )

    only_tables = select_relevant_tables(
        request.question, schema, max_tables=settings.schema_selection_max_tables
    )
    outcome = generate_and_execute_with_correction(
        question=request.question,
        schema=schema,
        llm=llm,
        engine=engine,
        only_tables=only_tables,
        max_attempts=settings.max_correction_attempts,
        max_rows=settings.max_rows,
        timeout_seconds=settings.query_timeout_seconds,
    )

    if outcome.result is None:
        return QueryResponse(
            question=request.question,
            sql=outcome.final_sql,
            error=outcome.error,
            attempts=outcome.attempt_count,
        )

    return QueryResponse(
        question=request.question,
        sql=outcome.final_sql,
        columns=outcome.result.columns,
        rows=outcome.result.rows,
        row_count=outcome.result.row_count,
        attempts=outcome.attempt_count,
        truncated=outcome.truncated,
    )