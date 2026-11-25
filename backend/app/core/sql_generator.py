"""
Question + schema -> SQL, via an LLMClient.

Step 2 gave this a naive, no-retry generate_sql(): it turns natural
language into a SQL string and never executes or retries anything itself.

Step 4 adds generate_corrected_sql(): the same job, but also told what SQL
was tried before and exactly why it failed (a database error, or this
agent's own SELECT-only guard), so the model can fix the specific problem
instead of guessing again from scratch. The self-correction loop that
calls it lives in app/core/correction_loop.py, not here — this module
stays limited to "turn a prompt into a SQL string", regardless of which
prompt.

Both prompts instruct the model to respond with ONLY a SQL statement, no
markdown fencing or commentary, but models don't always comply, so
extract_sql() strips common wrapping (```sql fences, a trailing semicolon)
defensively.
"""

from __future__ import annotations

import re

from app.core.errors import SQLGenerationError
from app.core.llm_client import LLMClient
from app.core.schema_introspector import SchemaSnapshot

SQL_GENERATION_PROMPT = """You are a SQL assistant. Given a database schema and a question \
in English, write a single SQL query that answers the question.

Rules:
- Use ONLY the tables and columns shown in the schema below. Never invent a table or column name.
- Write exactly one SQL statement. Do not include explanations, comments, or markdown formatting.
- Only SELECT statements are allowed. Never write INSERT, UPDATE, DELETE, DROP, or ALTER.

SCHEMA:
{schema}

QUESTION:
{question}

SQL:"""

CORRECTION_PROMPT = """You are a SQL assistant. A SQL query you previously wrote for the question \
below did not work. Write a corrected SQL query that fixes the problem and answers the question.

Rules:
- Use ONLY the tables and columns shown in the schema below. Never invent a table or column name.
- Write exactly one SQL statement. Do not include explanations, comments, or markdown formatting.
- Only SELECT statements are allowed. Never write INSERT, UPDATE, DELETE, DROP, or ALTER.

SCHEMA:
{schema}

QUESTION:
{question}

PREVIOUS SQL (this failed):
{failed_sql}

ERROR:
{error_message}

Write a corrected SQL query. Do not repeat the same mistake.

SQL:"""

_CODE_FENCE_RE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
# Matches stray markup such as "</comment>" that models sometimes emit.
_TAG_RE = re.compile(r"</?[A-Za-z][^>\n]*>")


def build_prompt(
    question: str, schema: SchemaSnapshot, only_tables: list[str] | None = None
) -> str:
    """Render the prompt sent to the LLM. Exposed separately so Step 4's
    self-correction loop can reuse it when building a retry prompt."""
    schema_text = schema.to_prompt_text(only_tables=only_tables)
    return SQL_GENERATION_PROMPT.format(schema=schema_text, question=question)


def build_correction_prompt(
    question: str,
    schema: SchemaSnapshot,
    failed_sql: str,
    error_message: str,
    only_tables: list[str] | None = None,
) -> str:
    """Render the retry prompt used by Step 4's self-correction loop,
    carrying forward the same table selection as the original attempt so
    the model sees a consistent view of the schema across retries."""
    schema_text = schema.to_prompt_text(only_tables=only_tables)
    return CORRECTION_PROMPT.format(
        schema=schema_text,
        question=question,
        failed_sql=failed_sql,
        error_message=error_message,
    )


def extract_sql(raw_response: str) -> str:
    """
    Turn a raw model reply into one SQL statement. Defensive because models
    don't always follow instructions:
      - strips markdown code fences
      - strips stray markup tags (e.g. "</comment>")
      - keeps only the first statement (anything after ";" is dropped, so a
        stacked "DROP TABLE" never reaches the database)
      - drops a repeated copy of the same statement on later lines
    Safety note: this is cleanup, not a security control. app.core.sql_safety
    is what actually blocks write statements, on every execution attempt.
    """
    cleaned = _CODE_FENCE_RE.sub("", raw_response)
    cleaned = _TAG_RE.sub("", cleaned).strip()
    cleaned = cleaned.split(";")[0].strip()

    kept: list[str] = []
    for line in cleaned.splitlines():
        if kept and line.strip().upper().startswith(("SELECT", "WITH")):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def generate_sql(
    question: str,
    schema: SchemaSnapshot,
    llm: LLMClient,
    only_tables: list[str] | None = None,
) -> str:
    """
    Question + schema -> SQL string. No execution, no retry — that is
    app.core.correction_loop's job. Raises SQLGenerationError if the model
    returns nothing usable.
    """
    prompt = build_prompt(question, schema, only_tables=only_tables)
    raw_response = llm.generate(prompt)
    sql = extract_sql(raw_response)
    if not sql:
        raise SQLGenerationError(f"LLM returned an empty SQL query for question: {question!r}")
    return sql


def generate_corrected_sql(
    question: str,
    schema: SchemaSnapshot,
    llm: LLMClient,
    failed_sql: str,
    error_message: str,
    only_tables: list[str] | None = None,
) -> str:
    """
    Like generate_sql(), but prompts with the previous attempt and its
    exact error so the model corrects the specific problem instead of
    generating blind. Raises SQLGenerationError if the model returns
    nothing usable — exactly like generate_sql(), so callers (the
    correction loop) can treat both the same way.
    """
    prompt = build_correction_prompt(
        question, schema, failed_sql, error_message, only_tables=only_tables
    )
    raw_response = llm.generate(prompt)
    sql = extract_sql(raw_response)
    if not sql:
        raise SQLGenerationError(
            f"LLM returned an empty corrected SQL query for question: {question!r}"
        )
    return sql