"""
Tests for Step 2's app/core/sql_generator.py.

Uses FakeLLMClient throughout — no real Gemini calls, no API key needed,
no cost, deterministic.
"""

import pytest
from sqlalchemy import create_engine, text

from app.core.errors import SQLGenerationError
from app.core.llm_client import FakeLLMClient
from app.core.schema_introspector import get_schema_snapshot
from app.core.sql_generator import (
    build_correction_prompt,
    build_prompt,
    extract_sql,
    generate_corrected_sql,
    generate_sql,
)


@pytest.fixture
def simple_schema():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE assets (id INTEGER PRIMARY KEY, name TEXT NOT NULL, "
                "criticality TEXT)"
            )
        )
    return get_schema_snapshot(engine)


def test_build_prompt_includes_schema_and_question(simple_schema):
    prompt = build_prompt("How many assets are critical?", simple_schema)
    assert "TABLE assets" in prompt
    assert "How many assets are critical?" in prompt


def test_build_prompt_respects_table_filter(simple_schema):
    prompt = build_prompt("question", simple_schema, only_tables=["assets"])
    assert "TABLE assets" in prompt


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("SELECT * FROM assets;", "SELECT * FROM assets"),
        ("```sql\nSELECT * FROM assets\n```", "SELECT * FROM assets"),
        ("```\nSELECT * FROM assets;\n```", "SELECT * FROM assets"),
        ("  SELECT * FROM assets  ", "SELECT * FROM assets"),
    ],
)
def test_extract_sql_strips_fences_and_whitespace(raw, expected):
    assert extract_sql(raw) == expected


def test_generate_sql_returns_cleaned_sql(simple_schema):
    llm = FakeLLMClient(response="```sql\nSELECT COUNT(*) FROM assets;\n```")
    sql = generate_sql("How many assets?", simple_schema, llm)
    assert sql == "SELECT COUNT(*) FROM assets"


def test_generate_sql_passes_question_and_schema_to_llm(simple_schema):
    llm = FakeLLMClient(response="SELECT 1")
    generate_sql("How many critical assets?", simple_schema, llm)
    assert len(llm.calls) == 1
    assert "How many critical assets?" in llm.calls[0]
    assert "TABLE assets" in llm.calls[0]


def test_generate_sql_raises_on_empty_response(simple_schema):
    llm = FakeLLMClient(response="   ")
    with pytest.raises(SQLGenerationError, match="empty SQL"):
        generate_sql("question", simple_schema, llm)


def test_generate_sql_can_use_responder_for_dynamic_answers(simple_schema):
    def responder(prompt: str) -> str:
        if "critical" in prompt:
            return "SELECT * FROM assets WHERE criticality = 'critical'"
        return "SELECT * FROM assets"

    llm = FakeLLMClient(responder=responder)
    sql = generate_sql("How many critical assets?", simple_schema, llm)
    assert "WHERE criticality = 'critical'" in sql

def test_extract_sql_removes_stray_tags_and_duplicate_statement():
    # Real malformed output seen from Gemini during a live /query test.
    raw = "SELECT count(*) FROM Assets</comment>\nSELECT count(*) FROM Assets</comment>"
    assert extract_sql(raw) == "SELECT count(*) FROM Assets"


def test_extract_sql_keeps_only_first_statement():
    assert extract_sql("SELECT a FROM t; DROP TABLE t;") == "SELECT a FROM t"


def test_extract_sql_keeps_multiline_select_intact():
    raw = "SELECT a\nFROM t\nWHERE a = 1"
    assert extract_sql(raw) == "SELECT a\nFROM t\nWHERE a = 1"

# --- Step 4: correction prompt ----------------------------------------------


def test_build_correction_prompt_includes_failed_sql_and_error(simple_schema):
    prompt = build_correction_prompt(
        "How many assets?",
        simple_schema,
        failed_sql="SELECT * FROM asets",
        error_message="no such table: asets",
    )
    assert "How many assets?" in prompt
    assert "SELECT * FROM asets" in prompt
    assert "no such table: asets" in prompt
    assert "TABLE assets" in prompt  # schema is still included


def test_build_correction_prompt_respects_table_filter(simple_schema):
    prompt = build_correction_prompt(
        "question",
        simple_schema,
        failed_sql="SELECT * FROM asets",
        error_message="no such table",
        only_tables=["assets"],
    )
    assert "TABLE assets" in prompt


def test_generate_corrected_sql_returns_cleaned_sql(simple_schema):
    llm = FakeLLMClient(response="```sql\nSELECT * FROM assets\n```")
    sql = generate_corrected_sql(
        "question",
        simple_schema,
        llm,
        failed_sql="SELECT * FROM asets",
        error_message="no such table",
    )
    assert sql == "SELECT * FROM assets"


def test_generate_corrected_sql_raises_on_empty_response(simple_schema):
    llm = FakeLLMClient(response="   ")
    with pytest.raises(SQLGenerationError, match="empty corrected SQL"):
        generate_corrected_sql(
            "question", simple_schema, llm, failed_sql="SELECT 1", error_message="some error"
        )