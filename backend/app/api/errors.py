"""
Maps exceptions to HTTP responses.

Every non-2xx response from this app has the same shape, matching FastAPI's
own convention so a frontend reads errors one way:

    {"detail": {"code": "<machine_readable>", "message": "<for humans>", ...}}

Rules:
  - Client-facing messages are fixed strings written here. Exception text
    (which can contain upstream provider detail) goes to the server log only.
  - Unexpected exceptions return a generic 500 and are logged with a
    traceback. Nothing internal reaches the client.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.errors import (
    AgentError,
    DatabaseUnavailableError,
    EmptySchemaError,
    LLMAuthError,
    LLMConfigurationError,
    LLMEmptyResponseError,
    LLMError,
    LLMQuotaExceededError,
    LLMUnavailableError,
    SQLGenerationError,
    UnsafeSQLError,
)

logger = logging.getLogger(__name__)

# (exception class, HTTP status, machine code, client message, log level).
# Order matters: first isinstance match wins, so subclasses come before LLMError.
_RULES: tuple[tuple[type[AgentError], int, str, str, int], ...] = (
    (
        LLMQuotaExceededError,
        429,
        "llm_quota_exceeded",
        "The language model's usage quota has been reached. Please try again later.",
        logging.WARNING,
    ),
    (
        LLMUnavailableError,
        503,
        "llm_unavailable",
        "The language model is temporarily unavailable. Please try again shortly.",
        logging.WARNING,
    ),
    (
        LLMAuthError,
        502,
        "llm_auth_error",
        "The language model rejected the server's credentials. "
        "This is a server configuration problem.",
        logging.ERROR,
    ),
    (
        LLMConfigurationError,
        503,
        "llm_not_configured",
        "The language model is not configured correctly on the server.",
        logging.ERROR,
    ),
    (
        LLMEmptyResponseError,
        502,
        "llm_bad_response",
        "The language model did not return an answer. Try rephrasing your question.",
        logging.WARNING,
    ),
    (
        SQLGenerationError,
        502,
        "sql_generation_failed",
        "Could not produce a SQL query for that question. Try rephrasing it.",
        logging.WARNING,
    ),
    (
        DatabaseUnavailableError,
        503,
        "database_unavailable",
        "The database is currently unavailable. Please try again shortly.",
        logging.ERROR,
    ),
    (
        EmptySchemaError,
        503,
        "schema_empty",
        "The connected database has no tables yet. Seed or migrate it before querying.",
        logging.WARNING,
    ),
    (
        # In normal operation Step 4's correction loop catches this itself
        # and never lets it reach here (see app/core/correction_loop.py).
        # This entry is a fail-safe: if it's ever raised somewhere that
        # loop doesn't wrap, the response is still a deliberate 400, not a
        # fall-through 500, and it's logged loudly enough to investigate.
        UnsafeSQLError,
        400,
        "unsafe_sql_blocked",
        "The generated query was not a safe, read-only SELECT statement and was blocked.",
        logging.ERROR,
    ),
    (
        LLMError,
        502,
        "llm_error",
        "The language model request failed.",
        logging.ERROR,
    ),
)


def _error_response(
    status_code: int,
    code: str,
    message: str,
    headers: dict[str, str] | None = None,
    **extra: object,
) -> JSONResponse:
    detail: dict[str, object] = {"code": code, "message": message, **extra}
    return JSONResponse(status_code=status_code, content={"detail": detail}, headers=headers)


async def handle_agent_error(request: Request, exc: Exception) -> JSONResponse:
    for exc_type, status_code, code, message, log_level in _RULES:
        if isinstance(exc, exc_type):
            logger.log(
                log_level,
                "%s on %s %s: %s",
                type(exc).__name__,
                request.method,
                request.url.path,
                exc,
            )
            if isinstance(exc, LLMQuotaExceededError) and exc.retry_after_seconds:
                return _error_response(
                    status_code,
                    code,
                    message,
                    headers={"Retry-After": str(exc.retry_after_seconds)},
                    retry_after_seconds=exc.retry_after_seconds,
                )
            return _error_response(status_code, code, message)

    # An AgentError subclass nobody mapped yet: fail safe, and say so in the log.
    logger.error(
        "Unmapped %s on %s %s: %s", type(exc).__name__, request.method, request.url.path, exc
    )
    return _error_response(500, "agent_error", "The request could not be completed.")


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled %s on %s %s",
        type(exc).__name__,
        request.method,
        request.url.path,
        exc_info=exc,
    )
    return _error_response(500, "internal_error", "An unexpected error occurred.")


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AgentError, handle_agent_error)
    app.add_exception_handler(Exception, handle_unexpected_error)
