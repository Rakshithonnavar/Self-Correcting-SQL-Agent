"""
Tests for Step 5's app/core/sql_safety.py — the full guardrail that runs
before every execution attempt in the correction loop: single-statement
read-only SELECT, every referenced table must exist in the connected
schema, and the result is capped at max_rows + 1.
"""

import pytest

from app.core.errors import UnsafeSQLError
from app.core.sql_safety import assert_safe_sql, dialect_for_engine

ALLOWED = {"assets", "vulnerabilities", "users"}


# --- Statement type -------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM assets",
        "select count(*) from assets",
        "  SELECT id FROM assets  ",
        "WITH recent AS (SELECT * FROM assets) SELECT * FROM recent",
        "-- a leading comment\nSELECT * FROM assets",
        "SELECT a.name FROM assets a JOIN vulnerabilities v ON a.id = v.asset_id",
    ],
)
def test_allows_select_and_with_statements(sql):
    assert_safe_sql(sql, ALLOWED, max_rows=100)  # must not raise


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
    with pytest.raises(UnsafeSQLError, match="read-only SELECT"):
        assert_safe_sql(sql, ALLOWED, max_rows=100)


def test_blocks_empty_sql():
    with pytest.raises(UnsafeSQLError, match="empty"):
        assert_safe_sql("", ALLOWED, max_rows=100)
    with pytest.raises(UnsafeSQLError, match="empty"):
        assert_safe_sql("   ", ALLOWED, max_rows=100)
    with pytest.raises(UnsafeSQLError, match="empty"):
        assert_safe_sql("-- only a comment", ALLOWED, max_rows=100)


def test_blocks_more_than_one_statement():
    with pytest.raises(UnsafeSQLError, match="Exactly one SQL statement"):
        assert_safe_sql("SELECT * FROM assets; DROP TABLE assets", ALLOWED, max_rows=100)


def test_blocks_unparsable_sql():
    with pytest.raises(UnsafeSQLError, match="could not be parsed"):
        assert_safe_sql("SELECT * FROM assets WHERE (", ALLOWED, max_rows=100)


# --- Table allowlist ---------------------------------------------------


def test_blocks_reference_to_a_table_outside_the_schema():
    with pytest.raises(UnsafeSQLError, match="not in the connected database's schema"):
        assert_safe_sql("SELECT * FROM sqlite_master", ALLOWED, max_rows=100)


def test_blocks_join_where_only_one_table_is_unknown():
    with pytest.raises(UnsafeSQLError, match="incidents"):
        assert_safe_sql(
            "SELECT * FROM assets JOIN incidents ON assets.id = incidents.asset_id",
            ALLOWED,
            max_rows=100,
        )


def test_allows_a_table_alias():
    assert_safe_sql("SELECT a.name FROM assets AS a", ALLOWED, max_rows=100)


def test_cte_defined_name_does_not_need_to_be_in_the_schema():
    # "recent" is defined by the WITH clause itself, not a real table, and
    # must not be treated as an unknown-table violation.
    assert_safe_sql(
        "WITH recent AS (SELECT * FROM assets) SELECT * FROM recent", ALLOWED, max_rows=100
    )


def test_table_name_comparison_is_case_insensitive():
    assert_safe_sql("SELECT * FROM ASSETS", ALLOWED, max_rows=100)
    assert_safe_sql("select * from assets", {"Assets"}, max_rows=100)


# --- Row-limit injection -------------------------------------------------


def test_adds_a_limit_when_the_query_has_none():
    rewritten = assert_safe_sql("SELECT * FROM assets", ALLOWED, max_rows=10)
    assert "LIMIT 11" in rewritten  # max_rows + 1, to detect truncation


def test_leaves_a_smaller_existing_limit_untouched():
    rewritten = assert_safe_sql("SELECT * FROM assets LIMIT 5", ALLOWED, max_rows=10)
    assert "LIMIT 5" in rewritten


def test_caps_an_existing_limit_above_the_budget():
    rewritten = assert_safe_sql("SELECT * FROM assets LIMIT 500", ALLOWED, max_rows=10)
    assert "LIMIT 11" in rewritten


def test_adds_limit_to_a_with_clause_query():
    rewritten = assert_safe_sql(
        "WITH recent AS (SELECT * FROM assets) SELECT * FROM recent", ALLOWED, max_rows=5
    )
    assert "LIMIT 6" in rewritten


# --- Dialect handling ---------------------------------------------------


def test_dialect_for_engine_maps_known_sqlalchemy_dialects():
    from sqlalchemy import create_engine

    sqlite_engine = create_engine("sqlite:///:memory:")
    assert dialect_for_engine(sqlite_engine) == "sqlite"


def test_result_is_re_serialized_in_the_requested_dialect():
    rewritten = assert_safe_sql("SELECT * FROM assets", ALLOWED, max_rows=10, dialect="sqlite")
    assert "assets" in rewritten.lower()


# --- Step 6 (Phase 2): known-value validation -------------------------------

KNOWN_VALUES = {
    "vulnerabilities": {"severity": ["critical", "high", "low", "medium"]},
    "assets": {"criticality": ["high", "low", "medium"]},
}


def test_allows_a_literal_that_matches_a_known_value():
    assert_safe_sql(
        "SELECT * FROM vulnerabilities WHERE severity = 'critical'",
        ALLOWED,
        max_rows=100,
        known_values=KNOWN_VALUES,
    )  # must not raise


def test_blocks_a_wrong_case_literal_against_a_known_value_column():
    with pytest.raises(UnsafeSQLError, match="'Critical' is not a value"):
        assert_safe_sql(
            "SELECT * FROM vulnerabilities WHERE severity = 'Critical'",
            ALLOWED,
            max_rows=100,
            known_values=KNOWN_VALUES,
        )


def test_error_message_names_the_real_valid_values():
    with pytest.raises(UnsafeSQLError, match="critical, high, low, medium"):
        assert_safe_sql(
            "SELECT * FROM vulnerabilities WHERE severity = 'Critical'",
            ALLOWED,
            max_rows=100,
            known_values=KNOWN_VALUES,
        )


def test_blocks_a_bad_literal_behind_a_table_alias():
    with pytest.raises(UnsafeSQLError, match="not a value"):
        assert_safe_sql(
            "SELECT * FROM vulnerabilities v WHERE v.severity = 'Critical'",
            ALLOWED,
            max_rows=100,
            known_values=KNOWN_VALUES,
        )


def test_blocks_a_bad_literal_inside_an_in_clause():
    with pytest.raises(UnsafeSQLError, match="not a value"):
        assert_safe_sql(
            "SELECT * FROM vulnerabilities WHERE severity IN ('low', 'Critical')",
            ALLOWED,
            max_rows=100,
            known_values=KNOWN_VALUES,
        )


def test_unqualified_column_is_resolved_across_an_unambiguous_join():
    # "severity" only exists in vulnerabilities' known_values, so this must
    # resolve correctly even though the column isn't qualified here.
    with pytest.raises(UnsafeSQLError, match="not a value"):
        assert_safe_sql(
            "SELECT * FROM assets a JOIN vulnerabilities v ON a.id = v.asset_id "
            "WHERE severity = 'Critical'",
            ALLOWED,
            max_rows=100,
            known_values=KNOWN_VALUES,
        )


def test_ambiguous_unqualified_column_is_not_checked():
    # Both tables have a "criticality"-shaped column in known_values here,
    # so an unqualified reference is ambiguous and must be skipped rather
    # than guessed - this must not raise, even with an unsampled casing.
    ambiguous_values = {
        "vulnerabilities": {"criticality": ["high", "low"]},
        "assets": {"criticality": ["high", "low", "medium"]},
    }
    assert_safe_sql(
        "SELECT * FROM assets JOIN vulnerabilities ON assets.id = vulnerabilities.asset_id "
        "WHERE criticality = 'Nonsense'",
        ALLOWED,
        max_rows=100,
        known_values=ambiguous_values,
    )  # must not raise: cannot tell which table's column this is


def test_column_with_no_known_values_is_not_checked():
    # "status" isn't in KNOWN_VALUES at all - any literal is allowed.
    assert_safe_sql(
        "SELECT * FROM vulnerabilities WHERE status = 'anything-goes'",
        ALLOWED,
        max_rows=100,
        known_values=KNOWN_VALUES,
    )  # must not raise


def test_known_values_defaults_to_no_validation():
    # Without known_values, the exact same bad-casing query is allowed -
    # confirms this check is strictly additive and opt-in.
    assert_safe_sql(
        "SELECT * FROM vulnerabilities WHERE severity = 'Critical'", ALLOWED, max_rows=100
    )  # must not raise
