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

from app.core.schema_introspector import get_schema_snapshot, populate_sample_values


def make_engine_with_schema(ddl_statements: list[str]):
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        for stmt in ddl_statements:
            conn.execute(text(stmt))
    return engine


@pytest.fixture
def two_table_engine():
    return make_engine_with_schema(
        [
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
        ]
    )


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
    engine = make_engine_with_schema(
        [
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
        ]
    )

    snapshot = get_schema_snapshot(engine)

    assert set(snapshot.table_names()) == {"customers", "orders", "order_items"}
    orders_fk = snapshot.tables["orders"].foreign_keys[0]
    assert orders_fk.references_table == "customers"
    items_fk = snapshot.tables["order_items"].foreign_keys[0]
    assert items_fk.references_table == "orders"


# --- Step 6: populate_sample_values -----------------------------------------


@pytest.fixture
def engine_with_data():
    """Same shape as two_table_engine, but seeded with real rows: a
    repeated, low-cardinality `criticality`/`severity` column (the kind
    Step 6 should hint), and a `title` column with as many distinct
    values as rows (the kind it should leave alone as free text)."""
    engine = make_engine_with_schema(
        [
            """
        CREATE TABLE assets (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            criticality TEXT
        )
        """,
            """
        CREATE TABLE vulnerabilities (
            id INTEGER PRIMARY KEY,
            asset_id INTEGER NOT NULL,
            severity TEXT NOT NULL,
            title TEXT,
            FOREIGN KEY (asset_id) REFERENCES assets(id)
        )
        """,
        ]
    )
    levels = ["low", "medium", "high", "critical"]
    with engine.begin() as conn:
        for i in range(1, 21):
            conn.execute(
                text("INSERT INTO assets (id, name, criticality) VALUES (:id, :name, :crit)"),
                {"id": i, "name": f"asset-{i}", "crit": levels[i % len(levels)]},
            )
        for i in range(1, 51):
            conn.execute(
                text(
                    "INSERT INTO vulnerabilities (id, asset_id, severity, title) "
                    "VALUES (:id, :asset_id, :sev, :title)"
                ),
                {
                    "id": i,
                    "asset_id": ((i - 1) % 20) + 1,
                    "sev": levels[i % len(levels)],
                    "title": f"unique finding #{i}",  # distinct every row
                },
            )
    return engine


def test_populate_sample_values_fills_low_cardinality_text_column(engine_with_data):
    schema = get_schema_snapshot(engine_with_data)
    populate_sample_values(engine_with_data, schema, max_distinct=12)
    crit_col = next(c for c in schema.tables["assets"].columns if c.name == "criticality")
    assert crit_col.sample_values == ["critical", "high", "low", "medium"]


def test_populate_sample_values_leaves_free_text_column_unsampled(engine_with_data):
    schema = get_schema_snapshot(engine_with_data)
    populate_sample_values(engine_with_data, schema, max_distinct=12)
    title_col = next(c for c in schema.tables["vulnerabilities"].columns if c.name == "title")
    assert title_col.sample_values is None


def test_populate_sample_values_never_samples_a_primary_key(engine_with_data):
    schema = get_schema_snapshot(engine_with_data)
    populate_sample_values(engine_with_data, schema, max_distinct=12)
    id_col = next(c for c in schema.tables["assets"].columns if c.name == "id")
    assert id_col.sample_values is None


def test_populate_sample_values_respects_only_tables(engine_with_data):
    schema = get_schema_snapshot(engine_with_data)
    populate_sample_values(engine_with_data, schema, only_tables=["assets"], max_distinct=12)
    crit_col = next(c for c in schema.tables["assets"].columns if c.name == "criticality")
    sev_col = next(c for c in schema.tables["vulnerabilities"].columns if c.name == "severity")
    assert crit_col.sample_values is not None
    assert sev_col.sample_values is None  # vulnerabilities was excluded from only_tables


def test_populate_sample_values_is_a_noop_without_any_data(two_table_engine):
    # Columns exist but no rows were ever inserted - must not raise, and
    # must leave every column unsampled (0 rows is not "fits within budget").
    schema = get_schema_snapshot(two_table_engine)
    populate_sample_values(two_table_engine, schema, max_distinct=12)
    for table in schema.tables.values():
        for column in table.columns:
            assert column.sample_values is None


def test_prompt_text_renders_sample_values_when_present(engine_with_data):
    schema = get_schema_snapshot(engine_with_data)
    populate_sample_values(engine_with_data, schema, max_distinct=12)
    prompt_text = schema.to_prompt_text()
    assert "criticality TEXT [values: critical, high, low, medium]" in prompt_text


def test_prompt_text_omits_value_hint_when_never_sampled(two_table_engine):
    # Sampling never ran against this schema - no column should render a hint.
    snapshot = get_schema_snapshot(two_table_engine)
    text_out = snapshot.to_prompt_text()
    assert "[values:" not in text_out
