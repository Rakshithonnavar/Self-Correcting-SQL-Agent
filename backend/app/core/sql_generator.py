"""
Step 2: question + schema -> SQL, via an LLMClient.

This is the naive, no-retry version (Step 2's "get a working question-
answer loop running"). Step 4 wraps this in a self-correction loop; this
module itself never retries and never executes anything — it only turns
natural language into a SQL string.

The prompt instructs the model to respond with ONLY a SQL statement, no
markdown fencing or commentary, but models don't always comply, so
extract_sql() strips common wrapping (```sql fences, stray tags, trailing
statements) defensively.
"""

from __future__ import annotations

import re

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


def extract_sql(raw_response: str) -> str:
    """
    Turn a raw model reply into one SQL statement. Defensive because models
    don't always follow instructions:
      - strips markdown code fences
      - strips stray markup tags (e.g. "</comment>")
      - keeps only the first statement (anything after ";" is dropped, so a
        stacked "DROP TABLE" never reaches the database)
      - drops a repeated copy of the same statement on later lines
    Safety note: this is cleanup, not a security control. Step 5's SELECT-only
    guardrail is what actually blocks write statements.
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
    Steps 4-6's job. Raises ValueError if the model returns nothing usable.
    """
    prompt = build_prompt(question, schema, only_tables=only_tables)
    raw_response = llm.generate(prompt)
    sql = extract_sql(raw_response)
    if not sql:
        raise ValueError(f"LLM returned an empty SQL query for question: {question!r}")
    return sql

