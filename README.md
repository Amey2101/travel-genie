# Travel Genie

AI-powered Travel Decision & Execution Agent — helps users make travel decisions and, where appropriate, execute them.

## Repository structure

```
travel-genie/
├── backend/          # Python backend (FastAPI + Pydantic)
│   ├── app/
│   │   ├── api/       # HTTP API layer (routes, request/response schemas)
│   │   ├── core/      # Cross-cutting concerns (config, security, logging)
│   │   ├── models/    # Domain / data models
│   │   ├── services/  # Application/business logic
│   │   ├── decision/  # Deterministic decision engine
│   │   ├── tools/     # Reusable capabilities / tools for the agent
│   │   └── providers/ # Replaceable external service interfaces
│   └── tests/         # Backend tests (pytest)
├── android/           # Android client (placeholder — not scaffolded yet)
└── docs/              # Project documentation
```

## Backend

A modular monolith. FastAPI provides the HTTP API; Pydantic provides schemas. No external integrations are configured yet.

### Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Run

```bash
cd backend
uvicorn app.main:app --reload
```

Health check: `GET http://127.0.0.1:8000/health`

### Test

```bash
cd backend
pytest
```

## Status

- [x] Repository scaffolding and backend project configuration
- [ ] Backend application logic
- [ ] Android client
- [ ] External integrations
