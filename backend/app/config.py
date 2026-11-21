"""
Centralised settings, read from environment variables (see .env.example).

Nothing elsewhere in the app should read os.environ directly — go through
Settings so there is exactly one place that knows where config comes from.
This is also what makes swapping SQLite for Postgres a one-line change:
DATABASE_URL is the only thing that differs between environments.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database: sqlite:///./dev.db locally, postgresql+psycopg2://... in prod (Neon)
    database_url: str = "sqlite:///./dev.db"

    # LLM
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # Agent behaviour
    max_self_correction_retries: int = 3
    query_timeout_seconds: int = 5
    max_result_rows: int = 1000

    # App
    environment: str = "development"  # development | production
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so Settings() is parsed from env only once per process."""
    return Settings()
