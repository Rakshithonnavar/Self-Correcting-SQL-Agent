"""
FastAPI app entrypoint.

Run locally (from backend/):
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_query
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

app.include_router(routes_query.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}