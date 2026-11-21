"""
Single source of truth for the SQLAlchemy engine.

Every other module (schema introspector, query executor, seed script)
imports get_engine() from here instead of creating its own connection.
That is what lets Step 0's introspection, and everything built on top of
it, work unchanged whether DATABASE_URL points at local SQLite or a
hosted Postgres (Neon) instance.
"""

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.config import get_settings


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    connect_args = {}
    if settings.database_url.startswith("sqlite"):
        # Needed for SQLite when the engine is shared across threads (FastAPI).
        connect_args = {"check_same_thread": False}
    return create_engine(settings.database_url, connect_args=connect_args)
