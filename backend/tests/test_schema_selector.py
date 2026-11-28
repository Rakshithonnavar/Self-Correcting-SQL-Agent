"""
Tests for Step 3's app/core/schema_selector.py.

Two kinds of coverage, deliberately kept separate:
  - Unit tests build a small SchemaSnapshot by hand, from a domain with
    no overlap with the demo database (customers/orders/products), to
    prove the selector carries no hardcoded table or column names.
  - One integration test runs it against the real seeded 8-table demo
    database, to show the behavior that actually ships: which tables a
    real question pulls in, and why.
"""

import pytest
from sqlalchemy import create_engine

from app.core.schema_introspector import (
    ColumnInfo,
    ForeignKeyInfo,
    SchemaSnapshot,
    TableInfo,
    get_schema_snapshot,
)
from app.core.schema_selector import select_relevant_tables
from app.db.seed_data import seed_database


def _col(name: str, type_: str = "TEXT", pk: bool = False) -> ColumnInfo:
    return ColumnInfo(name=name, type=type_, nullable=not pk, primary_key=pk)


@pytest.fixture
def shop_schema() -> SchemaSnapshot:
    """A small, unrelated schema (customers/orders/products/reviews) used
    only to prove the selector doesn't assume the demo database's names."""
    customers = TableInfo(
        name="customers",
        columns=[_col("id", pk=True), _col("full_name"), _col("signup_date")],
    )
    products = TableInfo(
        name="products",
        columns=[_col("id", pk=True), _col("title"), _col("price")],
    )
    orders = TableInfo(
        name="orders",
        columns=[_col("id", pk=True), _col("customer_id"), _col("order_date")],
        foreign_keys=[
            ForeignKeyInfo(
                column="customer_id", references_table="customers", references_column="id"
            )
        ],
    )
    reviews = TableInfo(
        name="reviews",
        columns=[_col("id", pk=True), _col("product_id"), _col("rating")],
        foreign_keys=[
            ForeignKeyInfo(column="product_id", references_table="products", references_column="id")
        ],
    )
    audit_log = TableInfo(name="audit_log", columns=[_col("id", pk=True), _col("event")])
    return SchemaSnapshot(
        tables={
            "customers": customers,
            "products": products,
            "orders": orders,
            "reviews": reviews,
            "audit_log": audit_log,
        }
    )


# --- Fallback-to-full-schema cases ------------------------------------------


def test_returns_none_when_max_tables_covers_whole_schema(shop_schema):
    assert select_relevant_tables("how many orders?", shop_schema, max_tables=5) is None
    assert select_relevant_tables("how many orders?", shop_schema, max_tables=10) is None


def test_returns_none_when_no_keyword_matches_any_table(shop_schema):
    result = select_relevant_tables("what is the meaning of life", shop_schema, max_tables=2)
    assert result is None


def test_returns_none_for_empty_or_punctuation_only_question(shop_schema):
    assert select_relevant_tables("", shop_schema, max_tables=2) is None
    assert select_relevant_tables("???", shop_schema, max_tables=2) is None


# --- Keyword matching ---------------------------------------------------


def test_selects_table_matching_a_question_keyword(shop_schema):
    result = select_relevant_tables("list all products", shop_schema, max_tables=2)
    assert result is not None
    assert "products" in result


def test_matches_are_schema_agnostic(shop_schema):
    """Same mechanism, unrelated domain — proves there's no hardcoded
    'assets'/'vulnerabilities' knowledge baked into the selector."""
    result = select_relevant_tables("show me customers", shop_schema, max_tables=2)
    assert result == ["customers"] or (result is not None and "customers" in result)


def test_singular_and_plural_question_words_both_match(shop_schema):
    singular = select_relevant_tables("find one customer", shop_schema, max_tables=2)
    plural = select_relevant_tables("find all customers", shop_schema, max_tables=2)
    assert singular is not None and "customers" in singular
    assert plural is not None and "customers" in plural


# --- Foreign-key expansion for joins ----------------------------------------


def test_includes_fk_connected_table_so_a_join_is_possible(shop_schema):
    # "orders" scores directly; "customers" only comes in via orders' own FK.
    result = select_relevant_tables("how many orders per customer", shop_schema, max_tables=4)
    assert result is not None
    assert "orders" in result
    assert "customers" in result


def test_reverse_fk_neighbor_is_included(shop_schema):
    # "products" scores directly; "reviews" references products, so a
    # question about products should still pull reviews in for a join.
    result = select_relevant_tables("average rating of products", shop_schema, max_tables=4)
    assert result is not None
    assert "products" in result
    assert "reviews" in result


def test_unrelated_table_is_excluded(shop_schema):
    result = select_relevant_tables("how many orders per customer", shop_schema, max_tables=4)
    assert result is not None
    assert "audit_log" not in result
    assert "products" not in result
    assert "reviews" not in result


# --- Budget is respected -----------------------------------------------------


def test_never_exceeds_max_tables(shop_schema):
    result = select_relevant_tables(
        "orders customers products reviews audit_log", shop_schema, max_tables=2
    )
    assert result is not None
    assert len(result) <= 2


def test_result_is_always_a_subset_of_schema_tables(shop_schema):
    result = select_relevant_tables("orders and customers and products", shop_schema, max_tables=3)
    assert result is not None
    assert set(result).issubset(set(shop_schema.table_names()))


def test_result_is_ordered_like_the_schema_not_by_score(shop_schema):
    # orders scores higher (direct keyword hit) than customers (FK-only),
    # but the returned order follows schema.table_names(), not score.
    result = select_relevant_tables("orders customer", shop_schema, max_tables=4)
    assert result is not None
    schema_order = shop_schema.table_names()
    assert result == [name for name in schema_order if name in result]


# --- Integration: the real demo database ------------------------------------


def test_real_demo_schema_selects_expected_tables_for_a_vulnerability_question():
    engine = create_engine("sqlite:///:memory:")
    seed_database(engine, seed=42)
    schema = get_schema_snapshot(engine)

    result = select_relevant_tables(
        "how many critical vulnerabilities are there", schema, max_tables=6
    )

    # "vulnerabilities" scores on its table name. "scan_findings" scores
    # too, but only because it has a vulnerability_id column — not because
    # of any hardcoded knowledge of this schema. "assets" and "scans" are
    # then pulled in as one-hop FK neighbors, so a join to either is still
    # possible. "users", "defects", "test_cases" and "test_runs" are
    # excluded: nothing in the question points at them, directly or via a
    # one-hop join.
    assert result == ["assets", "scan_findings", "scans", "vulnerabilities"]


def test_real_demo_schema_falls_back_to_full_schema_for_a_vague_question():
    engine = create_engine("sqlite:///:memory:")
    seed_database(engine, seed=42)
    schema = get_schema_snapshot(engine)

    result = select_relevant_tables("tell me something interesting", schema, max_tables=6)
    assert result is None
