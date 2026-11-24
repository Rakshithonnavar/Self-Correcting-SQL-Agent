"""
Step 2: POST /query — the basic text-to-SQL endpoint.

Naive version: question in, SQL out, SQL executed, result (or error) back.
No retry (Step 4), no guardrails (Step 5), no validation (Step 6) yet.
Those steps will modify what this endpoint calls, not its route shape.

Error handling: generate_sql() can raise LLMError subclasses (quota
exceeded, auth failure, provider unavailable, empty response) or
SQLGenerationError. None of those are caught here — they propagate to the
handlers registered in app/api/errors.py, which turn each into the right
HTTP status and a client-safe JSON body. Catching them per-route would mean
duplicating that mapping in every endpoint that calls the LLM.

QueryExecutionError (a failed/invalid generated SQL statement) is handled
differently and on purpose: Step 2 has no retry yet, so a bad query is a
normal, expected outcome — it's reported in the response body's `error`
field with a 200 status, not raised as an HTTP error. Step 4 will replace
this with the self-correction loop.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.engine import Engine

from app.core.llm_client import LLMClient
from app.core.query_executor import QueryExecutionError, execute_query
from app.core.schema_introspector import get_schema_snapshot
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
) -> QueryResponse:
    schema = get_schema_snapshot(engine)
    sql = generate_sql(request.question, schema, llm)

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