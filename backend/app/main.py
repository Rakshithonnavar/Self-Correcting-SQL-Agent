"""
FastAPI app entrypoint.

Run locally (from backend/):
    uvicorn app.main:app --reload --port 8000
(or let config.py's PORT setting decide, if you add a run script later)
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_query
from app.api.errors import register_exception_handlers
from app.config import get_settings

settings = get_settings()

app = FastAPI(title="Self-Correcting SQL Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Central error handling: anything raised as an app.core.errors.AgentError (or
# subclass) becomes a typed JSON error response (app/api/errors.py); anything
# else becomes a generic 500 with no internal detail leaked to the client.
register_exception_handlers(app)

app.include_router(routes_query.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}