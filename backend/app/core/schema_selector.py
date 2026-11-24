"""
Step 3: pick the subset of tables relevant to a question, instead of always
sending the whole schema to the LLM.

Why this exists: Step 0 made the agent schema-agnostic (point it at any
database and it introspects live). That's necessary but not sufficient for
a real org's database, which can have hundreds of tables. Sending all of
them on every question wastes tokens and money, adds latency, and gives the
model more irrelevant surface area to hallucinate a join across. This
module scores tables against the question's own words and keeps only the
ones that matter — no hardcoded table or column names, so it works the same
way against the demo database and against a schema it has never seen.

Design, deliberately simple and dependency-free (no embeddings, no vector
DB, no extra API calls — those would cost money and add a network
round-trip before we've even generated SQL):
  1. Tokenize the question and each table's name/column names into
     normalized word sets.
  2. Score each table by keyword overlap: a table-NAME match counts more
     than a column-name match, since "how many assets" is almost always
     about the `assets` table even before looking at its columns.
  3. Take the top-scoring tables, up to `max_tables`.
  4. Pull in anything directly foreign-key-connected to those tables (one
     hop, both directions), so the model can still write the join a
     question like "which assets have critical vulnerabilities" needs —
     scoring alone would only find `vulnerabilities`.
  5. If nothing scores, or the schema is already small enough that
     selecting a subset wouldn't shrink anything, return None: "send the
     full schema". Guessing a wrong subset is worse than sending extra
     tokens, so this never trims the schema on uncertain ground.

Public API:
    select_relevant_tables(question, schema, max_tables) -> list[str] | None
"""

from __future__ import annotations

import re

from app.core.schema_introspector import SchemaSnapshot, TableInfo

_WORD_RE = re.compile(r"[a-z0-9]+")

# Common words that appear in almost every question and would otherwise
# "match" against column names like `id`, `date`, `status` on every table,
# drowning out the words that actually identify which table is meant.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "where",
        "when",
        "why",
        "many",
        "much",
        "all",
        "any",
        "show",
        "me",
        "list",
        "give",
        "find",
        "get",
        "of",
        "in",
        "on",
        "at",
        "for",
        "to",
        "with",
        "and",
        "or",
        "by",
        "from",
        "that",
        "this",
        "there",
        "their",
        "it",
        "its",
        "per",
    }
)


def _normalize(token: str) -> str:
    """
    A deliberately crude singular/plural fold (e.g. "assets" -> "asset",
    "vulnerabilities" -> "vulnerability"). It doesn't need to be linguistically
    correct — only consistent — since it's applied identically to both the
    question and the schema's own names before comparing them.
    """
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _tokenize_question(question: str) -> set[str]:
    words = _WORD_RE.findall(question.lower())
    return {_normalize(w) for w in words if len(w) > 1 and w not in _STOPWORDS}


def _tokenize_identifier(identifier: str) -> set[str]:
    """Split a table or column name on underscores/non-word boundaries
    (e.g. "test_cases" -> {"test", "case"}, "asset_id" -> {"asset", "id"})."""
    parts = re.split(r"[_\W]+", identifier.lower())
    return {_normalize(p) for p in parts if p}


def _score_table(question_tokens: set[str], table: TableInfo) -> int:
    name_tokens = _tokenize_identifier(table.name)
    column_tokens: set[str] = set()
    for col in table.columns:
        column_tokens |= _tokenize_identifier(col.name)

    name_hits = len(question_tokens & name_tokens)
    column_hits = len(question_tokens & column_tokens)
    # A table-name match is a much stronger signal than a column-name match,
    # so it's weighted higher rather than treated as one more keyword hit.
    return 2 * name_hits + column_hits


def _fk_neighbors(table_name: str, schema: SchemaSnapshot) -> list[str]:
    """Tables directly reachable from `table_name` via a foreign key, in
    either direction, so a selected table can still be joined to what it
    references and to what references it."""
    table = schema.tables[table_name]
    neighbors = [fk.references_table for fk in table.foreign_keys]
    for other_name, other_table in schema.tables.items():
        if other_name == table_name:
            continue
        if any(fk.references_table == table_name for fk in other_table.foreign_keys):
            neighbors.append(other_name)

    seen: set[str] = set()
    ordered: list[str] = []
    for name in neighbors:
        if name not in seen and name in schema.tables:
            seen.add(name)
            ordered.append(name)
    return ordered


def select_relevant_tables(
    question: str,
    schema: SchemaSnapshot,
    max_tables: int = 6,
) -> list[str] | None:
    """
    Return the table names relevant to `question`, or None to mean "use
    the full schema" (either nothing scored, or the schema is already
    small enough that selecting wouldn't reduce anything).

    The returned list is ordered to match the schema's own table order
    (not by score), so the rendered prompt reads naturally and stays
    stable across calls for the same schema.
    """
    table_order = schema.table_names()
    if max_tables >= len(table_order):
        return None

    question_tokens = _tokenize_question(question)
    if not question_tokens:
        return None

    scores = {name: _score_table(question_tokens, schema.tables[name]) for name in table_order}
    scored_tables = sorted((n for n in table_order if scores[n] > 0), key=lambda n: -scores[n])
    if not scored_tables:
        return None

    selected: list[str] = list(scored_tables[:max_tables])

    # One-hop FK expansion so joins stay possible, without exceeding the budget.
    for name in list(selected):
        if len(selected) >= max_tables:
            break
        for neighbor in _fk_neighbors(name, schema):
            if len(selected) >= max_tables:
                break
            if neighbor not in selected:
                selected.append(neighbor)

    selected_set = set(selected)
    return [name for name in table_order if name in selected_set]
