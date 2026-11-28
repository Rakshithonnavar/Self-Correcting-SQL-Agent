"""
Domain exceptions for the SQL agent.

Why this module exists: the rest of the app must not depend on a specific
LLM SDK's exception types. GeminiClient (the only module that imports the
SDK) translates SDK/network errors into these, and the API layer
(app/api/errors.py) maps them to HTTP responses. Swapping providers later
then changes one translation function, not every caller.

Messages on these exceptions are for server logs. They may contain upstream
detail, so the API layer never sends them to clients verbatim.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for expected, domain-level failures."""


# --- LLM provider failures --------------------------------------------------


class LLMError(AgentError):
    """Base class for any failure talking to the LLM provider."""


class LLMConfigurationError(LLMError):
    """The LLM is not usable because of server configuration: missing API
    key, or a model name that doesn't exist / isn't enabled for the key."""


class LLMAuthError(LLMError):
    """The provider rejected the server's credentials (invalid or revoked
    key, or no permission)."""


class LLMQuotaExceededError(LLMError):
    """Rate limit or quota exhausted (HTTP 429 from the provider)."""

    def __init__(self, message: str, retry_after_seconds: int | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class LLMUnavailableError(LLMError):
    """Provider is down, overloaded, timed out, or unreachable. Retrying
    later may succeed."""


class LLMEmptyResponseError(LLMError):
    """The provider answered but returned no text (for example, the response
    was blocked by safety filters)."""


# --- Agent pipeline failures ------------------------------------------------


class SQLGenerationError(AgentError):
    """The model replied, but nothing usable could be extracted as SQL."""


class DatabaseUnavailableError(AgentError):
    """The target database could not be reached or inspected."""


class EmptySchemaError(AgentError):
    """The database was reached and introspected, but has no tables.
    Distinct from DatabaseUnavailableError: the connection is fine, there's
    just nothing to query yet (for example, before the demo data is seeded)."""


class UnsafeSQLError(AgentError):
    """Generated SQL failed Step 5's guardrails in app/core/sql_safety.py:
    it wasn't a single read-only SELECT/WITH statement, or it referenced a
    table outside the connected database's own schema. Blocked before it
    ever reached the database. In normal operation this is caught and
    corrected inside Step 4's self-correction loop
    (app/core/correction_loop.py) and never reaches the API layer; it's
    still mapped in app/api/errors.py as a fail-safe in case it's ever
    raised somewhere that loop doesn't wrap."""


class QueryTimeoutError(AgentError):
    """A generated query was still running after Step 5's execution
    timeout and was cancelled. Distinct from QueryExecutionError (the
    database rejected the query outright) and from LLMUnavailableError
    (the LLM, not the database, was slow): this means the database
    itself is taking too long on an otherwise-valid-looking query,
    usually because it scans far more data than the question needed."""
