"""
POST /query — the text-to-SQL endpoint.

Step 2 gave this a naive, no-retry version: question in, SQL out, SQL
executed, result (or error) back. No self-correction (Step 4), no
guardrails (Step 5), no validation (Step 6) yet — those steps will change
what this endpoint calls, not its route shape.

Step 3 adds smart schema selection: instead of handing the LLM every table
on every question, app/core/schema_selector.py scores tables against the
question's own words and keeps only the relevant ones (plus anything
directly joined to them). This matters once a real org's schema has
hundreds of tables — it doesn't change behavior noticeably on the 8-table
demo database, but it's exercised here the same way it would be in
production, and it's schema-agnostic like everything else in this agent.

Step 3 also adds a guard for an empty schema: a reachable but table-less
database (for example, before the demo data has been seeded) is a clear,
actionable error rather than letting the LLM guess at table names that
don't exist and surfacing a confusing SQL error instead.

Error handling: get_schema_snapshot() and generate_sql() can raise
EmptySchemaError, or LLMError subclasses (quota exceeded, auth failure,
provider unavailable, empty response), or SQLGenerationError. None of
those are caught here — they propagate to the handlers registered in
app/api/errors.py, which turn each into the right HTTP status and a
client-safe JSON body. Catching them per-route would mean duplicating
that mapping in every endpoint that calls the LLM.

QueryExecutionError (a failed/invalid generated SQL statement) is handled
differently and on purpose: there's no retry yet, so a bad query is a
normal, expected outcome — it's reported in the response body's `error`
field with a 200 status, not raised as an HTTP error. Step 4 will replace
this with the self-correction loop.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.engine import Engine

from app.config import Settings, get_settings
from app.core.errors import EmptySchemaError
from app.core.llm_client import LLMClient
from app.core.query_executor import QueryExecutionError, execute_query
from app.core.schema_introspector import get_schema_snapshot
from app.core.schema_selector import select_relevant_tables
from app.core.sql_generator import generate_sql
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
    sql = generate_sql(request.question, schema, llm, only_tables=only_tables)

    try:
        result = execute_query(sql, engine)
    except QueryExecutionError as exc:
        return QueryResponse(question=request.question, sql=sql, error=str(exc.original_error))

    return QueryResponse(
        question=request.question,
        sql=sql,
        columns=result.columns,
        rows=result.rows,
        row_count=result.row_count,
    )
