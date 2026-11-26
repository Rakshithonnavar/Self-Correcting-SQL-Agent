"""
Step 5: the full SQL safety guardrail.

Step 4 added a minimal, regex-based "does this start with SELECT" check,
added in a hurry because the self-correction loop had just started
executing SQL multiple times per request. This module replaces it with a
real one, built on a SQL parser (sqlglot) instead of a prefix match, so it
actually understands the statement instead of guessing from its first
word.

Three checks run in a single parse pass, because parsing is the most
expensive part of this and a generated query is already on the hot path
of every request:

  1. Exactly one statement, and it must be a SELECT (a WITH ... SELECT
     counts). Anything else - INSERT, UPDATE, DELETE, DROP, ALTER,
     CREATE, TRUNCATE, PRAGMA, ATTACH, or a second statement stacked
     after a ";" - is rejected outright. The prefix check extract_sql()
     already did (split on the first ";") made this hard to exploit even
     under the old regex, but a parser makes "hard" into "structurally
     impossible": there is no statement type a disguised prefix can hide
     behind, because we're reading what it actually parsed as.

  2. Every table the statement references must be a table that actually
     exists in the connected database's own introspected schema (passed
     in by the caller - never hardcoded here, keeping this schema-
     agnostic like the rest of the agent). This blocks a model from
     querying SQLite's own system catalog (sqlite_master), a table from
     a database it was never given, or anything else outside the schema
     it was shown - independent of whether that table happens to exist
     and the query would otherwise "work". A table a CTE defines for
     itself doesn't count as a reference and is excluded from this check.

  3. The result is row-limited by rewriting (not just checking) the
     statement's LIMIT clause, capped at max_rows + 1. The "+1" is
     deliberate: it lets the caller tell "exactly max_rows rows" apart
     from "more rows exist but were capped" without a second query. A
     query that already asks for fewer rows than the cap is left alone -
     capping only kicks in when there's no LIMIT, or the existing one
     exceeds the cap.

Scope: this is a single-database-at-a-time, read-only guardrail; it does
not attempt multi-tenant row-level security (which table a question is
even allowed to ask about, as opposed to which tables exist, is a future
concern once there's more than one demo dataset). It's also not yet a
full SQL-injection firewall against every exotic construct a dialect
supports - sqlglot's parser is the actual authority here, not a bespoke
blocklist, which is what makes this tractable at all.
"""

from __future__ import annotations

import sqlglot
from sqlalchemy.engine import Engine
from sqlglot import exp

from app.core.errors import UnsafeSQLError

# sqlglot's dialect names don't always match SQLAlchemy's; map the ones
# this project targets (SQLite locally, Postgres/Neon in production) so
# the SQL this module re-serializes always matches the engine it will
# actually run against.
_SQLGLOT_DIALECT_BY_SQLALCHEMY_NAME = {
    "sqlite": "sqlite",
    "postgresql": "postgres",
}


def dialect_for_engine(engine: Engine) -> str:
    """sqlglot dialect name for `engine`'s SQLAlchemy dialect, falling back
    to the SQLAlchemy name itself for anything not explicitly mapped."""
    name = engine.dialect.name
    return _SQLGLOT_DIALECT_BY_SQLALCHEMY_NAME.get(name, name)


def _referenced_tables(statement: exp.Select) -> set[str]:
    """Real table names the statement reads from, lowercased. Excludes
    names a WITH clause defines for itself (those are query-local aliases,
    not tables that need to exist in the schema)."""
    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    all_names = {table.name.lower() for table in statement.find_all(exp.Table)}
    return all_names - cte_names


def _cap_row_limit(statement: exp.Select, max_rows: int) -> None:
    """Mutates `statement` in place so it never returns more than
    max_rows + 1 rows. A smaller existing LIMIT is left untouched."""
    existing = statement.args.get("limit")
    existing_value: int | None = None
    if existing is not None:
        try:
            existing_value = int(existing.expression.this)
        except (AttributeError, TypeError, ValueError):
            existing_value = None  # a non-literal LIMIT; treat as uncapped

    if existing_value is None or existing_value > max_rows:
        statement.set("limit", exp.Limit(expression=exp.Literal.number(max_rows + 1)))


def assert_safe_sql(
    sql: str,
    allowed_tables: set[str],
    max_rows: int,
    dialect: str = "sqlite",
) -> str:
    """
    Validate `sql` and return the (possibly row-capped) statement that is
    safe to execute. Raises UnsafeSQLError, without ever touching the
    database, if `sql`:
      - doesn't parse as valid SQL in `dialect`,
      - isn't exactly one statement,
      - isn't a SELECT (or WITH ... SELECT), or
      - references a table not in `allowed_tables`.
    """
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise UnsafeSQLError(f"Generated SQL could not be parsed: {exc}") from exc

    if len(statements) == 0:
        raise UnsafeSQLError("Generated SQL was empty or contained no statement.")
    if len(statements) > 1:
        raise UnsafeSQLError(f"Exactly one SQL statement is allowed; got {len(statements)}.")

    statement = statements[0]
    if not isinstance(statement, exp.Select):
        raise UnsafeSQLError(
            "Only read-only SELECT statements are allowed; got a "
            f"{type(statement).__name__} statement."
        )

    allowed = {name.lower() for name in allowed_tables}
    unknown = _referenced_tables(statement) - allowed
    if unknown:
        raise UnsafeSQLError(
            "References table(s) not in the connected database's schema: "
            f"{', '.join(sorted(unknown))}."
        )

    _cap_row_limit(statement, max_rows)
    return statement.sql(dialect=dialect)