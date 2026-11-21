"""
Tests for Step 0 (app/core/schema_introspector.py).

These exist to prove the one claim the whole architecture depends on:
nothing here hardcodes a table or column name. Every assertion below
reads values that came from the schema we created in the fixture, never
a name baked into the introspector itself. If someone later "fixes" a
failing test by hardcoding a name in the source, these tests should start
failing against a *different* schema — see test_introspection_is_schema_agnostic.
"""

import pytest
from sqlalchemy import create_engine, text

from app.core.schema_introspector import get_schema_snapshot


def make_engine_with_schema(ddl_statements: list[str]):
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        for stmt in ddl_statements:
            conn.execute(text(stmt))
    return engine


@pytest.fixture
def two_table_engine():
    return make_engine_with_schema([
        """
        CREATE TABLE assets (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            environment TEXT,
            criticality TEXT
        )
        """,
        """
        CREATE TABLE vulnerabilities (
            id INTEGER PRIMARY KEY,
            asset_id INTEGER NOT NULL,
            severity TEXT NOT NULL,
            status TEXT,
            FOREIGN KEY (asset_id) REFERENCES assets(id)
        )
        """,
    ])


def test_detects_all_tables(two_table_engine):
    snapshot = get_schema_snapshot(two_table_engine)
    assert set(snapshot.table_names()) == {"assets", "vulnerabilities"}


def test_detects_columns_and_types(two_table_engine):
    snapshot = get_schema_snapshot(two_table_engine)
    assets = snapshot.tables["assets"]
    names = {c.name for c in assets.columns}
    assert names == {"id", "name", "environment", "criticality"}


def test_detects_primary_key(two_table_engine):
    snapshot = get_schema_snapshot(two_table_engine)
    assets = snapshot.tables["assets"]
    assert assets.primary_key_columns == ["id"]
    id_col = next(c for c in assets.columns if c.name == "id")
    assert id_col.primary_key is True


def test_detects_not_null(two_table_engine):
    snapshot = get_schema_snapshot(two_table_engine)
    assets = snapshot.tables["assets"]
    name_col = next(c for c in assets.columns if c.name == "name")
    env_col = next(c for c in assets.columns if c.name == "environment")
    assert name_col.nullable is False
    assert env_col.nullable is True


def test_detects_foreign_key(two_table_engine):
    snapshot = get_schema_snapshot(two_table_engine)
    vulns = snapshot.tables["vulnerabilities"]
    assert len(vulns.foreign_keys) == 1
    fk = vulns.foreign_keys[0]
    assert fk.column == "asset_id"
    assert fk.references_table == "assets"
    assert fk.references_column == "id"


def test_prompt_text_includes_foreign_keys(two_table_engine):
    snapshot = get_schema_snapshot(two_table_engine)
    text_out = snapshot.to_prompt_text()
    assert "FOREIGN KEYS: asset_id -> assets.id" in text_out


def test_prompt_text_can_be_filtered_to_subset(two_table_engine):
    snapshot = get_schema_snapshot(two_table_engine)
    filtered = snapshot.to_prompt_text(only_tables=["assets"])
    assert "TABLE assets" in filtered
    assert "TABLE vulnerabilities" not in filtered


def test_to_dict_round_trips_structure(two_table_engine):
    snapshot = get_schema_snapshot(two_table_engine)
    as_dict = snapshot.to_dict()
    assert as_dict["dialect"] == "sqlite"
    assert set(as_dict["tables"].keys()) == {"assets", "vulnerabilities"}
    assert as_dict["tables"]["assets"]["primary_key_columns"] == ["id"]


def test_introspection_is_schema_agnostic():
    """
    The real test of Step 0's promise: point it at a *completely different*
    schema (different names, different table count, different FK direction)
    and confirm it still works without touching the introspector's source.
    """
    engine = make_engine_with_schema([
        """
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            email TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            total_cents INTEGER NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
        """,
        """
        CREATE TABLE order_items (
            item_id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            sku TEXT NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        )
        """,
    ])

    snapshot = get_schema_snapshot(engine)

    assert set(snapshot.table_names()) == {"customers", "orders", "order_items"}
    orders_fk = snapshot.tables["orders"].foreign_keys[0]
    assert orders_fk.references_table == "customers"
    items_fk = snapshot.tables["order_items"].foreign_keys[0]
    assert items_fk.references_table == "orders"
