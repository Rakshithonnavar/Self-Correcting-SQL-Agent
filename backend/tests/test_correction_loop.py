"""
Tests for Step 4's app/core/correction_loop.py.

FakeLLMClient's `responder` callable lets each test script exactly what
the model "says" on each successive call, which is how a correction
sequence (fail, then fail differently, then succeed) is simulated without
any real API calls.
"""

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from app.core.correction_loop import generate_and_execute_with_correction
from app.core.errors import (
    LLMQuotaExceededError,
    LLMUnavailableError,
    SQLGenerationError,
)
from app.core.llm_client import FakeLLMClient
from app.core.schema_introspector import get_schema_snapshot


@pytest.fixture
def seeded_engine():
    # StaticPool matters here (not just for the FastAPI TestClient case):
    # execute_query_with_timeout runs the query in a worker thread, and a
    # plain sqlite:///:memory: engine hands a brand-new, empty database to
    # each new thread's connection. StaticPool forces every checkout to
    # reuse the one real connection, so the worker thread sees the same
    # seeded data the test set up.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE assets (id INTEGER PRIMARY KEY, name TEXT NOT NULL)"))
        conn.execute(text("INSERT INTO assets (id, name) VALUES (1, 'payments-api')"))
    return engine


@pytest.fixture
def schema(seeded_engine):
    return get_schema_snapshot(seeded_engine)


def test_succeeds_on_first_attempt_without_any_correction(seeded_engine, schema):
    llm = FakeLLMClient(response="SELECT * FROM assets")
    outcome = generate_and_execute_with_correction(
        "show me all assets", schema, llm, seeded_engine, max_attempts=3
    )

    assert outcome.error is None
    assert outcome.result is not None
    assert outcome.result.row_count == 1
    assert outcome.attempt_count == 1
    assert len(llm.calls) == 1  # no correction call made


def test_corrects_a_bad_table_name_on_the_second_attempt(seeded_engine, schema):
    responses = iter(
        [
            "SELECT * FROM asets",  # typo: wrong table name, will fail
            "SELECT * FROM assets",  # corrected
        ]
    )

    def responder(_prompt: str) -> str:
        return next(responses)

    llm = FakeLLMClient(responder=responder)
    outcome = generate_and_execute_with_correction(
        "show me all assets", schema, llm, seeded_engine, max_attempts=3
    )

    assert outcome.error is None
    assert outcome.result is not None
    assert outcome.result.row_count == 1
    assert outcome.attempt_count == 2
    assert outcome.attempts[0].error is not None
    assert outcome.attempts[1].error is None
    # The retry prompt must carry the previous SQL and its real error, so
    # the model can see what specifically went wrong. "asets" is caught by
    # Step 5's table-allowlist check (it isn't a table in the schema) before
    # it ever reaches the database, so this is now the guardrail's own
    # message rather than SQLite's "no such table" error.
    assert "asets" in llm.calls[1]
    assert "not in the connected database's schema" in llm.calls[1]


def test_blocks_and_corrects_an_unsafe_write_statement(seeded_engine, schema):
    responses = iter(
        [
            "DELETE FROM assets",  # unsafe: must never execute
            "SELECT * FROM assets",  # corrected to something safe
        ]
    )

    def responder(_prompt: str) -> str:
        return next(responses)

    llm = FakeLLMClient(responder=responder)
    outcome = generate_and_execute_with_correction(
        "remove nothing, just show assets", schema, llm, seeded_engine, max_attempts=3
    )

    assert outcome.error is None
    assert outcome.result is not None
    assert outcome.attempt_count == 2
    assert "read-only SELECT" in outcome.attempts[0].error

    # The unsafe statement must never have reached the database: the row
    # inserted in the fixture is still the only one, nothing was deleted.
    with seeded_engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM assets")).scalar() == 1


def test_gives_up_after_max_attempts_and_reports_the_last_error(seeded_engine, schema):
    llm = FakeLLMClient(response="SELECT * FROM table_that_does_not_exist")
    outcome = generate_and_execute_with_correction(
        "anything", schema, llm, seeded_engine, max_attempts=3
    )

    assert outcome.result is None
    assert outcome.error is not None
    assert "table_that_does_not_exist" in outcome.error
    assert outcome.attempt_count == 3
    assert len(llm.calls) == 3  # one initial generation + two correction retries


def test_never_executes_an_unsafe_statement_even_as_the_final_attempt(seeded_engine, schema):
    llm = FakeLLMClient(response="DROP TABLE assets")
    outcome = generate_and_execute_with_correction(
        "anything", schema, llm, seeded_engine, max_attempts=2
    )

    assert outcome.result is None
    assert "read-only SELECT" in outcome.error
    with seeded_engine.connect() as conn:
        # If this raises, the table was actually dropped — the worst-case
        # failure this guard exists to prevent.
        assert conn.execute(text("SELECT COUNT(*) FROM assets")).scalar() == 1


def test_llm_provider_error_is_not_retried_and_propagates_immediately(seeded_engine, schema):
    call_count = 0

    def responder(_prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        raise LLMQuotaExceededError("quota exhausted", retry_after_seconds=60)

    llm = FakeLLMClient(responder=responder)
    with pytest.raises(LLMQuotaExceededError):
        generate_and_execute_with_correction("anything", schema, llm, seeded_engine, max_attempts=3)

    # Must fail fast on the very first call, not burn through retries on a
    # provider failure that unrestricted retrying would only repeat.
    assert call_count == 1


def test_llm_error_on_a_correction_retry_also_propagates_immediately(seeded_engine, schema):
    calls = {"n": 0}

    def responder(_prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "SELECT * FROM asets"  # fails, triggers a correction call
        raise LLMUnavailableError("upstream is down")

    llm = FakeLLMClient(responder=responder)
    with pytest.raises(LLMUnavailableError):
        generate_and_execute_with_correction("anything", schema, llm, seeded_engine, max_attempts=3)

    assert calls["n"] == 2


def test_sql_generation_error_is_not_retried(seeded_engine, schema):
    llm = FakeLLMClient(response="   ")  # extracts to empty SQL
    with pytest.raises(SQLGenerationError):
        generate_and_execute_with_correction("anything", schema, llm, seeded_engine, max_attempts=3)
    assert len(llm.calls) == 1


def test_rejects_max_attempts_below_one(seeded_engine, schema):
    llm = FakeLLMClient(response="SELECT * FROM assets")
    with pytest.raises(ValueError, match="at least 1"):
        generate_and_execute_with_correction("anything", schema, llm, seeded_engine, max_attempts=0)


def test_schema_and_only_tables_are_not_recomputed_between_attempts(seeded_engine, schema):
    """The correction loop takes schema/only_tables as given — it must not
    call back into introspection or selection itself. This is enforced by
    construction (the loop has no engine-introspection import), and this
    test is the behavioral proof: passing a narrowed-down, single-table
    only_tables list still lets correction succeed using that same table."""
    responses = iter(["SELECT * FROM asets", "SELECT * FROM assets"])

    def responder(_prompt: str) -> str:
        return next(responses)

    llm = FakeLLMClient(responder=responder)
    outcome = generate_and_execute_with_correction(
        "show assets", schema, llm, seeded_engine, only_tables=["assets"], max_attempts=3
    )

    assert outcome.error is None
    assert "TABLE assets" in llm.calls[0]
    assert "TABLE assets" in llm.calls[1]


# --- Step 5: guardrails wired into the loop ---------------------------------


def test_corrects_a_query_referencing_a_table_outside_the_schema(seeded_engine, schema):
    """The table allowlist check (unlike the old regex guard) distinguishes
    'this table doesn't exist in the database at all' from an ordinary bad
    table name - both are still correctable the same way."""
    responses = iter(
        [
            "SELECT * FROM incidents",  # not in the schema at all
            "SELECT * FROM assets",  # corrected
        ]
    )

    def responder(_prompt: str) -> str:
        return next(responses)

    llm = FakeLLMClient(responder=responder)
    outcome = generate_and_execute_with_correction(
        "show me all incidents", schema, llm, seeded_engine, max_attempts=3
    )

    assert outcome.error is None
    assert outcome.attempt_count == 2
    assert "not in the connected database's schema" in outcome.attempts[0].error
    # The correction prompt must name the offending table so the model can
    # actually fix it, not just see a generic "it failed".
    assert "incidents" in llm.calls[1]


def test_query_outside_the_narrowed_only_tables_is_not_rejected_as_unsafe(seeded_engine, schema):
    """only_tables (Step 3) only shrinks what's SHOWN to the model in the
    prompt - it must never become a security boundary. A query the model
    writes against a real schema table that merely fell outside that
    narrowed subset must still be allowed to execute."""
    llm = FakeLLMClient(response="SELECT * FROM assets")
    outcome = generate_and_execute_with_correction(
        "show assets",
        schema,
        llm,
        seeded_engine,
        only_tables=["some_other_table"],  # deliberately doesn't include "assets"
        max_attempts=3,
    )
    assert outcome.error is None
    assert outcome.result is not None


def test_truncates_results_over_max_rows_and_reports_it(seeded_engine, schema):
    with seeded_engine.begin() as conn:
        for i in range(2, 6):
            conn.execute(
                text("INSERT INTO assets (id, name) VALUES (:id, :name)"),
                {"id": i, "name": f"asset-{i}"},
            )

    llm = FakeLLMClient(response="SELECT * FROM assets")
    outcome = generate_and_execute_with_correction(
        "show all assets", schema, llm, seeded_engine, max_attempts=1, max_rows=3
    )

    assert outcome.error is None
    assert outcome.truncated is True
    assert outcome.result is not None
    assert outcome.result.row_count == 3
    assert len(outcome.result.rows) == 3


def test_does_not_report_truncated_when_under_the_limit(seeded_engine, schema):
    llm = FakeLLMClient(response="SELECT * FROM assets")
    outcome = generate_and_execute_with_correction(
        "show all assets", schema, llm, seeded_engine, max_attempts=1, max_rows=100
    )
    assert outcome.truncated is False
    assert outcome.result.row_count == 1


def test_corrects_a_query_that_times_out(seeded_engine, schema):
    """A genuinely slow query (a real recursive CTE, not mocked) is
    cancelled and treated as correctable, exactly like a bad table name."""
    responses = iter(
        [
            "WITH RECURSIVE cnt(x) AS "
            "(SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 500000000) "
            "SELECT COUNT(*) FROM cnt",
            "SELECT * FROM assets",  # corrected to something fast
        ]
    )

    def responder(_prompt: str) -> str:
        return next(responses)

    llm = FakeLLMClient(responder=responder)
    outcome = generate_and_execute_with_correction(
        "anything",
        schema,
        llm,
        seeded_engine,
        max_attempts=2,
        timeout_seconds=0.3,
    )

    assert outcome.error is None
    assert outcome.attempt_count == 2
    assert "did not complete" in outcome.attempts[0].error