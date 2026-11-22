"""
Step 2: POST /query — the basic text-to-SQL endpoint.

Naive version: question in, SQL out, SQL executed, result (or error) back.
No retry (Step 4), no guardrails (Step 5), no validation (Step 6) yet.
Those steps will modify what this endpoint calls, not its route shape.

Note: kept at /query (no /api prefix). The repo's current README mentions
/api/query, but that README is being treated as placeholder content for
now (per instruction) and will be redefined later — not followed here.
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