"""
Tests for Step 4's app/core/sql_safety.py — the minimal SELECT-only guard
that runs before every execution attempt in the correction loop.
"""

import pytest

from app.core.errors import UnsafeSQLError
from app.core.sql_safety import ensure_select_only


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM assets",
        "select count(*) from assets",
        "  SELECT id FROM assets  ",
        "WITH recent AS (SELECT * FROM assets) SELECT * FROM recent",
        "-- a leading comment\nSELECT * FROM assets",
    ],
)
def test_allows_select_and_with_statements(sql):
    ensure_select_only(sql)  # must not raise


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE assets",
        "DELETE FROM assets",
        "UPDATE assets SET name = 'x'",
        "INSERT INTO assets (id, name) VALUES (1, 'x')",
        "ALTER TABLE assets ADD COLUMN x TEXT",
        "CREATE TABLE evil (id INTEGER)",
        "TRUNCATE TABLE assets",
        "PRAGMA table_info(assets)",
        "ATTACH DATABASE 'x.db' AS x",
    ],
)
def test_blocks_write_and_admin_statements(sql):
    with pytest.raises(UnsafeSQLError, match="not a read-only SELECT"):
        ensure_select_only(sql)


def test_blocks_empty_sql():
    with pytest.raises(UnsafeSQLError, match="was empty"):
        ensure_select_only("")
    with pytest.raises(UnsafeSQLError, match="was empty"):
        ensure_select_only("   ")


def test_select_disguised_as_a_later_statement_is_still_blocked():
    # Defense-in-depth: even if something upstream failed to isolate the
    # first statement, a write statement as the FIRST thing is blocked
    # regardless of what follows it.
    with pytest.raises(UnsafeSQLError):
        ensure_select_only("DROP TABLE assets; SELECT * FROM assets")