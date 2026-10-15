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
│   │   ├── decision/  # Deterministic decision engine (implementation)
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

A modular monolith. FastAPI provides the HTTP API; Pydantic provides schemas. The static web frontend is served by the same application. Destination discovery is live (SerpApi Google Maps). Requirement extraction is wired to real LLM providers (Google Gemini and Groq) behind the `LLMClient` contract; provider selection is explicit via `LLM_PROVIDER`, defaulting to a credential-free `stub`. Once requirements are complete the agent runs a search and a deterministic decision step; the search provider is selected explicitly via `SEARCH_PROVIDER` and defaults to a clearly-labelled synthetic demo provider (no live bookable inventory exists yet).

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

- `POST /workflows`, `POST /workflows/{id}/messages`, `GET /workflows/{id}` — conversational requirement collection, followed by search + deterministic recommendation once requirements are complete.
- `POST /workflows/{id}/requirements` — deterministic structured-input entry point used by the home-page form. It maps validated form fields into the same extraction/clarification/search/decision pipeline as the text path (no second orchestrator). A stated budget is a soft target (`approximate`), never a hard maximum.
- `POST /workflows/{id}/clarifications` — deterministic clarification answers (from/to, dates/duration, travelers, budget). It merges the supplied values into the current extraction and resumes the same pipeline, so clarifications are answerable with the default `stub` provider (which cannot parse free text). `409` when there is nothing to clarify yet, `422` for inconsistent answers.
- `GET /api/discover?destination=...&category=...` — live destination discovery (SerpApi Google Maps): places to see and do, not bookable inventory. Returns `503 discovery_unavailable` when no `SERPAPI_API_KEY` is configured.
- `POST /api/plan` — **retained compatibility placeholder**: always returns `501 not_implemented`. The frontend no longer calls it; no parallel planner is implemented.
- `GET /` and `GET /discover` — static pages; assets under `/static`.

#### Frontend

The static home page (`backend/static/app.js`) drives the `TravelAgent` workflow directly: it creates a workflow, submits the structured form to `/workflows/{id}/requirements`, renders clarification questions (answered with a deterministic form posting to `/workflows/{id}/clarifications`) and the returned recommendation, renders the deterministic `budget_estimate` as a separate panel (per-category estimates, total, target comparison and suggestions, with an explicit "not verified prices or bookings" disclaimer), and labels synthetic demo results as non-bookable. It never fabricates prices, availability, itineraries or bookings, and it does not call `/api/plan`.

### Configuration

Canonical config is the repository-root `.env` (single source; `backend/.env` is not read). Copy `.env.example` to `.env` and fill in values as integrations are added. Values already exported in the shell take precedence over the file. Keys are read at call time and are never logged or returned by the API. Real secret values are never printed during validation.

### LLM providers (requirement extraction)

Requirement extraction supports three selected providers; selection is explicit and never falls back automatically.

| `LLM_PROVIDER` | Backend | Credentials required | Notes |
| --- | --- | --- | --- |
| `stub` (default) | `StubLLMClient` | none | Deterministic local responses; used for offline runs and tests |
| `gemini` | Google Gemini (`generateContent` REST) | `GEMINI_API_KEY` | Model: `GEMINI_MODEL` (default `gemini-3.8-flash`) |
| `groq` | Groq (OpenAI-compatible `/chat/completions`) | `GROQ_API_KEY` | Model: `GROQ_MODEL` (default `openai/gpt-oss-120b`) |

- Providers send the extraction system prompt plus the target JSON Schema and validate the returned JSON against the typed `LLMExtractionOutput` model; deterministic transformation then drives the `TravelAgent` as before.
- `GEMINI_TIMEOUT_SECONDS` / `GROQ_TIMEOUT_SECONDS` bound each HTTP call; `GEMINI_MAX_OUTPUT_TOKENS` / `GROQ_MAX_OUTPUT_TOKENS` cap output.
- Run offline: leave `LLM_PROVIDER` unset (or `stub`) — the app runs without any API credentials.
- Configure a live provider: set `LLM_PROVIDER=gemini` (or `groq`) and the matching key. Missing credentials produce a controlled workflow failure (`llm_not_configured`) on the first extraction call — the server still starts and the health/discovery endpoints still work.
- Run tests: the test suite uses mocked HTTP and requires no credentials.

### Planning pipeline (search + deterministic decision)

When extraction completes with no open **required** clarification, `TravelAgent` runs the planning step on the same workflow (no second agent, no parallel workflow):

```
validated requirements
    -> SearchProvider.search(SearchRequest)         # bookable-inventory boundary
    -> deterministic hard-constraint filtering
    -> deterministic ranking + explanations
    -> Recommendation persisted on AgentState
    -> returned as AgentResponse.recommendation (+ GET /workflows/{id})
```

- **Hard constraints first.** `HardConstraints` (`max_budget`, `required_transport`, `min/max_duration_days`, `departure_time`, `arrival_time`, `date_range`) are enforced before scoring. A constraint that cannot be evaluated against a candidate's data is never assumed to pass: the candidate is rejected with an explicit machine-readable reason (e.g. an unknown price under a hard max budget is `max_budget_unverified`, not silently accepted).
- **Ranking is deterministic and transparent.** Feasible candidates are scored from available data and the existing soft-preference schema (budget fit, preferred transport, preferred departure time), with a stable `provider_ref` tie-break. Unknown metrics are never imputed, and the LLM is never used for filtering, scoring, approval, or execution.
- **Outcomes are explicit.** `Recommendation.status` is one of `ok`, `no_results`, `no_feasible_options`, `provider_unavailable`, `decision_error`. A provider failure is never disguised as a successful empty search, and never yields a 500.

#### Demo search provider (synthetic, not bookable)

`SEARCH_PROVIDER=demo` (the default) selects `DemoSearchProvider`. It is deterministic (SHA-256-derived fixtures, stable across runs) and makes no network calls. **Every** result is labelled `source="demo"` and `bookable=false`; synthetic prices are a function of the route only and never read the user's budget. It is not a `DiscoveryProvider` and does not reuse Google Maps attractions as transport/accommodation inventory.

There is **no real flight/hotel inventory provider and no booking** in this build. The demo provider exists only to exercise the pipeline locally and in tests; it can be replaced through the same `SearchProvider` interface. Provider selection is explicit and there is no automatic fallback to demo data after a live provider fails.

### Budget estimation (deterministic, informational)

Whenever the planning step returns a recommendation, `TravelAgent` runs a deterministic budget step on the **same** workflow (still one orchestrator) and exposes it as `AgentResponse.budget_estimate` / `AgentState.budget_estimate`. The estimate can therefore also accompany `no_results` / `no_feasible_options` outcomes; a provider or decision **exception** returns before this step and leaves no estimate.

```
recommendation (ok, no_results, or no_feasible_options)
    -> budget_input_from_requirements(requirements, recommendation)
    -> StyleBasedBudgetEstimator.estimate(input)
    -> BudgetEstimate persisted on AgentState
    -> returned as AgentResponse.budget_estimate (+ GET /workflows/{id})
```

- **Deterministic, no LLM, no network.** `app/decision/budget.py` adapts the teammate's rule-of-thumb `heuristic_lines` / `compute_budget` helpers (fixed per-style cost tables) into typed models. Identical input always yields identical output. Budget estimation is deliberately separate from ranking: it never influences filtering, scoring, approval, or execution.
- **Estimates are not prices.** `estimated_total` is always the exact sum of the category lines. The `note` and `disclaimer` state explicitly that these are planning estimates, **not verified provider prices**, and that nothing has been booked.
- **Budget semantics are preserved.** An approximate target (`budget.total` with `is_approximate=true`) is never treated as a hard ceiling; a hard maximum (`hard_constraints.max_budget`) is surfaced separately as `hard_max_budget`. An absent budget yields `target_budget=null` and an explicit reference-only note rather than a fabricated comparison.
- **Provenance is explicit.** `transportation_source` is `heuristic` (a rule-of-thumb assumption) or `provider_verified` (only when a selected candidate's price is genuinely `source="verified"`). Synthetic demo prices are **never** reused as real fares. `budget_input_from_requirements` returns `None` when no meaningful estimate is possible (missing timing, traveller count, or an unsupported currency), and the estimate is then simply absent — never invented; unsupported currency codes are never treated as 1:1 with INR.
- **Date/duration interpretation.** A concrete `date_range` is counted inclusively (return date is a travel day: `days = end - start + 1`, `nights = days - 1`); otherwise a stated `duration_days` is used (`nights = days - 1`). This is a deliberate estimator policy and does not change existing extraction/clarification semantics.

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
- [x] Real LLM providers (Gemini and Groq) behind `LLMClient` for structured requirement extraction
- [x] Planning pipeline (validated requirements → search → deterministic recommendation), with an explicit synthetic demo `SearchProvider`
- [x] Decision engine: deterministic hard-constraint filtering, ranking, explanations, typed outcomes
- [x] Frontend/workflow integration: the home page drives `TravelAgent` via a deterministic structured-input adapter and renders real recommendations (`/api/plan` remains an unused 501 placeholder)
- [x] Deterministic budget estimation integrated into the `TravelAgent` workflow and rendered as a separate estimate panel (clearly distinguished from verified prices)
- [ ] Real bookable transport/accommodation inventory search (`SearchProvider`)
- [ ] Itinerary generation (only ranked options plus a budget estimate are shown)
- [ ] Execution provider (booking, cancellation)
- [ ] MongoDB persistence
- [ ] Android client
