"""
Step 0: Dynamic schema introspection.

Nothing in this module (or anywhere downstream) hardcodes a table or column
name. The schema is read from the live database on every call, so pointing
this agent at a different database requires only a new connection string.

Public API:
    get_schema_snapshot(engine) -> SchemaSnapshot
    SchemaSnapshot.to_prompt_text() -> str   (compact text for an LLM prompt)
    SchemaSnapshot.to_dict() -> dict         (for a future Schema Explorer UI)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import inspect
from sqlalchemy.engine import Engine


@dataclass
class ColumnInfo:
    name: str
    type: str
    nullable: bool
    primary_key: bool


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

    def to_prompt_text(self, only_tables: Optional[list[str]] = None) -> str:
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
                col_parts.append(f"{col.name} {col.type}{tag_str}")
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
