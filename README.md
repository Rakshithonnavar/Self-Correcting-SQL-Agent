# Self-Correcting SQL Agent

A natural-language-to-SQL application that generates SQL with Gemini, executes it safely, retries after SQL errors, and exposes audit/evaluation information.

## Features

- Gemini-powered SQL generation
- SQLite and PostgreSQL support
- Schema introspection and relevant-table selection
- SELECT-only guardrails
- Statement timeout and result row limits
- Self-correction retries
- Structured audit logging
- Fixed evaluation dataset and report endpoint
- React/Vite dashboard

## Quick start

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Set `GEMINI_API_KEY` in `backend/.env` to enable LLM SQL generation. The API remains usable with a deterministic fallback generator when no key is configured.

Seed and inspect the database:

```bash
python -m app.db.seed_data
curl http://localhost:8000/health
```

API documentation: http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for the request flow and component responsibilities.

## API

- `POST /api/query` — answer a natural-language question
- `GET /api/audit-log` — inspect query attempts
- `GET /api/eval-report` — run or retrieve evaluation summary
- `GET /health` — service health

## Safety

The backend rejects non-SELECT statements, multiple statements, SQL comments, and high-risk DDL/DML keywords before execution. Results are capped and queries run through SQLAlchemy text execution with a SQLite timeout and row limit.

## Known limitations

See [docs/limitations.md](docs/limitations.md).
