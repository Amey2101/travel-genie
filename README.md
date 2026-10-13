# Travel Genie

AI-powered Travel Decision & Execution Agent — helps users make travel decisions and, where appropriate, execute them.

## Repository structure

```
travel-genie/
├── backend/          # Python backend (FastAPI + Pydantic)
│   ├── app/
│   │   ├── agent/     # Deterministic orchestrator, state machine, security contracts
│   │   ├── api/       # HTTP API layer (routes, error envelope, schemas)
│   │   ├── core/      # Cross-cutting concerns (config, logging)
│   │   ├── decision/  # Deterministic decision engine (contracts only)
│   │   ├── models/    # Domain / data models
│   │   ├── services/  # Application/business logic
│   │   ├── tools/     # Tool/provider protocols (search, execution)
│   │   └── providers/ # Replaceable external service interfaces
│   ├── static/        # Static web frontend (HTML/CSS/JS)
│   └── tests/         # Backend tests (pytest)
├── android/           # Android client (placeholder — not scaffolded yet)
└── docs/              # Project documentation
```

## Backend

A modular monolith. FastAPI provides the HTTP API; Pydantic provides schemas. The static web frontend is served by the same application. SerpApi live destination discovery is wired for the Explore page; Gemini/Groq LLM clients are configured but not yet called, so the conversational requirement agent runs against a stub LLM.

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

Open <http://127.0.0.1:8000> for the web UI. Health checks:

- `GET /health` → `{"status": "ok"}`
- `GET /api/health` → setup summary (`mode`, `env_file_found`, per-integration `configured` flags); never returns key values.

Endpoints:

- `POST /workflows`, `POST /workflows/{id}/messages`, `GET /workflows/{id}` — conversational requirement collection.
- `GET /api/discover?destination=...&category=...` — live destination discovery (SerpApi Google Maps): places to see and do, not bookable inventory. Returns `503 discovery_unavailable` when no `SERPAPI_API_KEY` is configured.
- `POST /api/plan` — compatibility endpoint for the frontend planner; returns `501 not_implemented` until the planning pipeline lands.
- `GET /` and `GET /discover` — static pages; assets under `/static`.

### Configuration

Canonical config is the repository-root `.env` (single source; `backend/.env` is not read). Copy `.env.example` to `.env` and fill in values as integrations are added. Values already exported in the shell take precedence over the file. Keys are read at call time and are never logged or returned by the API.

### Test

```bash
cd backend
pytest
```

## Status

- [x] Repository scaffolding and backend project configuration
- [x] Backend scaffold (health endpoint, domain model, LLM requirement extraction)
- [x] Agent orchestration & clarification state machine
- [x] Application foundation integration (single app, config, static frontend, error envelope)
- [x] Destination discovery integration (SerpApi Google Maps behind `DiscoveryProvider`)
- [ ] Real LLM providers (Gemini primary, Groq fallback) behind `LLMClient`
- [ ] Planning pipeline (itinerary generation, deterministic budget, fallback)
- [ ] Bookable transport/accommodation inventory search (`SearchProvider`)
- [ ] Decision engine scoring and ranking
- [ ] Execution provider (booking, cancellation)
- [ ] MongoDB persistence
- [ ] Android client
