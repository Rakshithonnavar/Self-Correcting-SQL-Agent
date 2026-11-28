"""
Step 0: Dynamic schema introspection.

Nothing in this module (or anywhere downstream) hardcodes a table or column
name. The schema is read from the live database on every call, so pointing
this agent at a different database requires only a new connection string.

Step 6 adds populate_sample_values(): a follow-up pass over an already-
introspected schema that samples the real distinct values of low-
cardinality text columns (severity, status, criticality, ...), so the
prompt shows the model what's actually stored instead of leaving it to
guess a plausible-looking value ('Critical' when the data holds
'critical'). This is a prompt-quality improvement, not a safety control -
app.core.sql_safety still decides what's safe to execute.

Public API:
    get_schema_snapshot(engine) -> SchemaSnapshot
    populate_sample_values(engine, schema, only_tables=None, max_distinct=12)
    SchemaSnapshot.to_prompt_text() -> str   (compact text for an LLM prompt)
    SchemaSnapshot.to_dict() -> dict         (for a future Schema Explorer UI)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


@dataclass
class ColumnInfo:
    name: str
    type: str
    nullable: bool
    primary_key: bool
    # Step 6: the column's real distinct values, filled in by
    # populate_sample_values() for low-cardinality text columns only.
    # None means "not sampled" (numeric/date/binary type, too many
    # distinct values to be categorical, or sampling never ran) - it is
    # never an empty list, so callers can treat "truthy" as "has a hint".
    sample_values: list[str] | None = None


@dataclass
class ForeignKeyInfo:
    column: str
    references_table: str
    references_column: str


@dataclass
class TableInfo:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    foreign_keys: list[ForeignKeyInfo] = field(default_factory=list)
    primary_key_columns: list[str] = field(default_factory=list)


@dataclass
class SchemaSnapshot:
    tables: dict[str, TableInfo] = field(default_factory=dict)
    dialect: str = ""

    def table_names(self) -> list[str]:
        return list(self.tables.keys())

    def to_prompt_text(self, only_tables: list[str] | None = None) -> str:
        """
        Render the schema as compact text for an LLM prompt.
        If only_tables is given (see Step 3's schema selector), render just
        those tables instead of the whole database.
        """
        names = only_tables if only_tables is not None else self.table_names()
        lines = []
        for name in names:
            table = self.tables.get(name)
            if table is None:
                continue
            col_parts = []
            for col in table.columns:
                tag = []
                if col.primary_key:
                    tag.append("PK")
                if col.nullable is False:
                    tag.append("NOT NULL")
                tag_str = f" ({', '.join(tag)})" if tag else ""
                values_str = ""
                if col.sample_values:
                    values_str = f" [values: {', '.join(col.sample_values)}]"
                col_parts.append(f"{col.name} {col.type}{tag_str}{values_str}")
            fk_parts = [
                f"{fk.column} -> {fk.references_table}.{fk.references_column}"
                for fk in table.foreign_keys
            ]
            block = f"TABLE {name} (\n  " + ",\n  ".join(col_parts) + "\n)"
            if fk_parts:
                block += "\nFOREIGN KEYS: " + "; ".join(fk_parts)
            lines.append(block)
        return "\n\n".join(lines)

    def to_dict(self) -> dict:
        """Structured form, for a future Schema Explorer UI (deferred feature)."""
        return {
            "dialect": self.dialect,
            "tables": {
                name: {
                    "columns": [
                        {
                            "name": c.name,
                            "type": c.type,
                            "nullable": c.nullable,
                            "primary_key": c.primary_key,
                            "sample_values": c.sample_values,
                        }
                        for c in t.columns
                    ],
                    "foreign_keys": [
                        {
                            "column": fk.column,
                            "references_table": fk.references_table,
                            "references_column": fk.references_column,
                        }
                        for fk in t.foreign_keys
                    ],
                    "primary_key_columns": t.primary_key_columns,
                }
                for name, t in self.tables.items()
            },
        }


def get_schema_snapshot(engine: Engine) -> SchemaSnapshot:
    """
    Read the live schema from `engine` via SQLAlchemy's inspector.
    Works identically against SQLite and PostgreSQL (and anything else
    SQLAlchemy supports) because it never touches a hardcoded name.
    """
    inspector = inspect(engine)
    snapshot = SchemaSnapshot(dialect=engine.dialect.name)

    for table_name in inspector.get_table_names():
        pk_constraint = inspector.get_pk_constraint(table_name)
        pk_columns = pk_constraint.get("constrained_columns", []) or []

        columns = []
        for col in inspector.get_columns(table_name):
            columns.append(
                ColumnInfo(
                    name=col["name"],
                    type=str(col["type"]),
                    nullable=col.get("nullable", True),
                    primary_key=col["name"] in pk_columns,
                )
            )

        foreign_keys = []
        for fk in inspector.get_foreign_keys(table_name):
            referred_table = fk.get("referred_table")
            referred_cols = fk.get("referred_columns", []) or []
            local_cols = fk.get("constrained_columns", []) or []
            for local_col, ref_col in zip(local_cols, referred_cols):
                foreign_keys.append(
                    ForeignKeyInfo(
                        column=local_col,
                        references_table=referred_table,
                        references_column=ref_col,
                    )
                )

        snapshot.tables[table_name] = TableInfo(
            name=table_name,
            columns=columns,
            foreign_keys=foreign_keys,
            primary_key_columns=pk_columns,
        )

    return snapshot


# --- Step 6: sample-value hints ---------------------------------------------

# Prefixes of SQLAlchemy's str(column.type) output worth sampling as
# categorical hints. Deliberately excludes numeric, date/time, boolean, and
# binary/blob types: a numeric or date column has no fixed vocabulary to
# hint, and sampling one would just waste a query.
_TEXTUAL_TYPE_PREFIXES = (
    "VARCHAR",
    "NVARCHAR",
    "CHAR",
    "NCHAR",
    "TEXT",
    "CLOB",
    "STRING",
    "ENUM",
)


def _is_textual_type(type_str: str) -> bool:
    return type_str.upper().startswith(_TEXTUAL_TYPE_PREFIXES)


def populate_sample_values(
    engine: Engine,
    schema: SchemaSnapshot,
    only_tables: list[str] | None = None,
    max_distinct: int = 12,
) -> None:
    """
    Fill in ColumnInfo.sample_values for low-cardinality text columns, by
    querying the live database for each candidate column's real distinct
    values. Mutates `schema` in place.

    only_tables narrows which tables are probed - pass the same
    only_tables Step 3's schema selector already computed for this
    request, so the number of queries this runs scales with what's
    actually going in the prompt, not the whole database. Pass None to
    probe every table in the schema.

    Only textual columns are probed (see _is_textual_type), and primary
    keys are skipped outright - an id column is never categorical, and
    probing it would just waste a query that's guaranteed to exceed
    max_distinct. For each remaining column this runs one query:
    "SELECT DISTINCT col FROM table WHERE col IS NOT NULL LIMIT
    max_distinct + 1". If more rows come back than max_distinct, the
    column holds free text (e.g. a title, a description, an email) rather
    than a fixed set of values, and is left with sample_values=None -
    showing the model an arbitrary, truncated slice of a genuinely
    unbounded column would be more misleading than no hint at all.

    Security: every identifier this builds into SQL - table and column
    names - is quoted through this engine's own
    dialect.identifier_preparer, and originates only from the schema
    passed in (itself produced moments earlier by SQLAlchemy's inspector
    in get_schema_snapshot), never from user input or LLM output. The
    LIMIT value is passed as a bound parameter, not interpolated.

    Performance, stated plainly rather than hidden: LIMIT bounds how many
    rows come back, not how much work the database does to produce them -
    "SELECT DISTINCT" on a column with no supporting index can still scan
    the whole table. That is an acceptable cost at this project's demo
    scale (thousands of rows), and is exactly why this feature is gated
    behind settings.schema_sample_values_enabled: it can be switched off
    instantly against a larger or unindexed production schema without
    touching code. A failure probing one column (for example, a type this
    dialect filters differently than expected) is caught and that column
    is simply left unsampled, rather than failing the whole request over
    what is a prompt-quality enhancement, not a correctness guarantee.
    """
    quote = engine.dialect.identifier_preparer.quote
    table_names = only_tables if only_tables is not None else schema.table_names()

    with engine.connect() as conn:
        for table_name in table_names:
            table = schema.tables.get(table_name)
            if table is None:
                continue
            for column in table.columns:
                if column.primary_key or not _is_textual_type(column.type):
                    continue
                try:
                    rows = conn.execute(
                        text(
                            f"SELECT DISTINCT {quote(column.name)} FROM {quote(table_name)} "
                            f"WHERE {quote(column.name)} IS NOT NULL LIMIT :limit"
                        ),
                        {"limit": max_distinct + 1},
                    ).fetchall()
                except SQLAlchemyError:
                    continue  # best-effort hint; never fails the request over it

                if 0 < len(rows) <= max_distinct:
                    column.sample_values = sorted(str(row[0]) for row in rows)
