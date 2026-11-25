"""
Minimal SQL safety guard, introduced in Step 4 and expanded in Step 5.

Why this exists now rather than waiting for Step 5: Step 4's self-
correction loop (app/core/correction_loop.py) executes generated SQL
multiple times per request instead of once. That widens the window for a
write statement to slip through unnoticed if the model ever drifts from the
"only SELECT statements" instruction in its prompt — a prompt is guidance,
not enforcement. This module is the enforcement: a lightweight, fail-closed
check that runs before every execution attempt, on the first attempt and on
every correction retry alike.

Scope, on purpose: this checks that the statement READS AS a SELECT/WITH
and rejects everything else (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE,
TRUNCATE, PRAGMA, ATTACH, and anything else). It is a prefix check, not a
full SQL parser, and it does not yet enforce a row cap or a statement
timeout. Step 5 is where this becomes the complete guardrail set the
architecture calls for — likely including a real SQL parser (e.g.
sqlglot) instead of a regex, for queries too irregular for a prefix check
to classify correctly.
"""

from __future__ import annotations

import re

from app.core.errors import UnsafeSQLError

# Matches optional leading SQL line-comments, then the statement's first
# keyword. Only SELECT and WITH (the start of a read-only CTE) pass.
_READ_ONLY_PREFIX_RE = re.compile(r"^(?:\s*--[^\n]*\n)*\s*(select|with)\b", re.IGNORECASE)


def ensure_select_only(sql: str) -> None:
    """
    Raise UnsafeSQLError unless `sql` reads as a single read-only
    SELECT/WITH statement. Called by the correction loop before every
    execution attempt — never bypassed, including on the very first try.
    """
    if not sql.strip():
        raise UnsafeSQLError("Generated SQL was empty.")
    if not _READ_ONLY_PREFIX_RE.match(sql):
        raise UnsafeSQLError(
            "Generated SQL is not a read-only SELECT statement and was blocked before execution."
        )
