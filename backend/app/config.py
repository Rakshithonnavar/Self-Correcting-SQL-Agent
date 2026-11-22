"""
Centralised settings, read from environment variables (see .env.example).

Nothing elsewhere in the app should read os.environ directly — go through
Settings so there is exactly one place that knows where config comes from.
This is also what makes swapping SQLite for Postgres a one-line change:
DATABASE_URL is the only thing that differs between environments.

Field names match the .env/.env.example variable names exactly (pydantic-
settings maps field_name -> FIELD_NAME by default), so what you set in
.env is actually read — nothing here silently falls back to a default
because of a name mismatch.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM
    gemini_api_key: str = ""
    # "gemini-2.5-flash" is a well-established, currently-available model as
    # of this writing. Model availability and free-tier limits change and
    # differ per account — check https://aistudio.google.com for what your
    # key can use, and override via GEMINI_MODEL in .env if needed.
    gemini_model: str = "gemini-2.5-flash"

    # Database: sqlite:///./sql_agent.db locally, postgresql+psycopg2://... in prod (Neon)
    database_url: str = "sqlite:///./sql_agent.db"

    # Server
    port: int = 8000
    cors_origins: str = "http://localhost:5173"

    # App
    log_level: str = "INFO"

    # Agent behaviour
    max_rows: int = 100
    query_timeout_seconds: int = 10
    max_correction_attempts: int = 3

    def cors_origin_list(self) -> list[str]:
        """CORS_ORIGINS is a single comma-separated string in .env; split it here."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so Settings() is parsed from env only once per process."""
    return Settings()