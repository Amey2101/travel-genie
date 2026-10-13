# AGENTS.md

## Current state

Verified facts (re-investigate before relying on this file):

- Backend scaffold: FastAPI + Pydantic app with `GET /health`, `pyproject.toml` (setuptools), and pytest tests in `backend/`.
- Domain model: `TravelRequirements` with extraction pipeline types (`ExtractionResult`, `ClarificationItem`).
- LLM boundary: `LLMClient` Protocol, `StubLLMClient`, `RequirementExtractionService` (structured extraction + deterministic transform). Live providers: `GeminiLLMClient` (`app/providers/gemini_llm.py`, Groq `app/providers/groq_llm.py`; both synchronous `httpx` REST calls). Provider selection is explicit via `LLM_PROVIDER` (`stub` default | `gemini` | `groq`), selected by `build_llm_client()` in `app/providers/factory.py` and injected in `app/main.py:create_app`. No automatic fallback. Provider failures map to controlled errors (`LLMProviderError` with `llm_not_configured`/`llm_request_failed`/`llm_invalid_response`); missing credentials fail the workflow without a 500 or leaking secrets.
- Agent layer: `TravelAgent` orchestrator, typed `WorkflowStatus` state machine, deterministic `policy` (gap detection), approval security boundary (`app/agent/security.py`). Once requirements are complete (no open required clarification) the agent runs the planning step in the same workflow: `SearchProvider` → deterministic `DecisionEngine`, persisting `AgentState.search_results`/`recommendation` and returning `AgentResponse.recommendation`.
- Planning: `DeterministicDecisionEngine` (`app/decision/deterministic.py`) implements the `DecisionEngine` contract — hard-constraint filtering before ranking, transparent deterministic scoring (budget/transport/departure-time, stable `provider_ref` tie-break), machine-readable rejection reasons, and typed outcomes (`ok`/`no_results`/`no_feasible_options`/`provider_unavailable`/`decision_error`). Unknown metrics are never imputed; no LLM is used for filtering/scoring/approval/execution. Search provider is selected explicitly via `SEARCH_PROVIDER` (`demo` default) through `build_search_provider()` in `app/providers/factory.py`; no automatic fallback.
- `DemoSearchProvider` (`app/providers/demo_search.py`): deterministic synthetic fixtures behind `SearchProvider` (SHA-256-derived, no network); every result is `source="demo"` and `bookable=false`, prices depend only on the route (never the user's budget), and it is deliberately not a `DiscoveryProvider`.
- Contracts: `SearchProvider` (demo implementation only; no live inventory), `DecisionEngine` (deterministic implementation), `ExecutionProvider` (contract only; no provider/implementation).
- Workflow API: `POST /workflows`, `POST /workflows/{id}/messages`, `POST /workflows/{id}/requirements`, `POST /workflows/{id}/clarifications`, `GET /workflows/{id}`.
- Frontend integration (Milestone 5): `backend/static/app.js` drives `TravelAgent` (creates a workflow, submits to `/workflows/{id}/requirements`, answers clarifications deterministically via `/workflows/{id}/clarifications` — `merge_clarification` in `app/services/structured_requirements.py` resumes the shared pipeline so the `stub` provider needs no NL parsing — renders `AgentResponse.recommendation` and marks synthetic results non-bookable). It no longer calls `/api/plan`. The structured form is mapped by `app/services/structured_requirements.py` (deterministic adapter reusing `to_extraction_result`); `TravelAgent.handle_structured` shares the exact turn logic with the text path (one orchestrator). Form budget is always `approximate`, never a hard max. Browser-level JS is not exercised by tests (only served-asset assertions).
- Foundation integration (Milestone 1): single app in `app/main.py` serves the static frontend (`backend/static/`, routes `/` and `/discover`, assets under `/static`); env config ported to `app/core/config.py` (`python-dotenv` runtime dep); unified `{"error": {...}}` envelope in `app/api/errors.py`; `GET /health` preserved and `GET /api/health` setup summary added; `POST /api/plan` is a `501 not_implemented` compatibility placeholder (retained, unused by the frontend).
- Discovery integration (Milestone 2): `DiscoveryProvider` protocol in `app/tools/discovery.py`; `SerpApiDiscoveryProvider` adapter in `app/providers/serpapi_discovery.py` (SerpApi Google Maps, `httpx` runtime dep); `GET /api/discover` (destination places/attractions, not bookable inventory) with validation, 503 `discovery_unavailable` when unconfigured, bounded timeouts, sanitized errors, http(s)-only result links. `SearchProvider` (bookable inventory) remains a contract only and is deliberately separate from discovery. Canonical config is repo-root `.env` (single source; shell exports win).
- No real bookable transport/accommodation inventory providers (OpenRouter deferred; the only `SearchProvider` is the synthetic `DemoSearchProvider`; `ExecutionProvider` remains a contract only), no execution, no MongoDB, no Android.
- `android/`, `docs/` are placeholders.
- Remote is `https://github.com/Amey2101/travel-genie.git`, working on `main`.

Backend commands (run from `backend/` after `pip install -e ".[dev]"`):

- Run: `uvicorn app.main:app --reload`
- Test: `pytest`

## Product

Travel Genie is an AI-powered Travel Decision & Execution Agent — not a generic travel chatbot. It helps users make travel decisions and, where appropriate, execute them.

## Architecture

### Backend
- Python
- FastAPI
- Pydantic
- MongoDB Atlas
- Modular monolith

### Android client
- Kotlin
- Jetpack Compose
- MVVM
- Retrofit
- Keep business logic and secrets on the backend.

### AI responsibilities
- LLMs handle natural-language understanding, structured requirement extraction, reasoning, explanation, and orchestration where appropriate.
- Deterministic Python code handles validation, calculations, constraint filtering, scoring, ranking, and other deterministic business logic.
- External-world information must come from tools/providers/APIs rather than LLM knowledge.
- Use structured outputs and strict schemas.

### Agent architecture
- Start with a single orchestrator.
- Do not introduce multiple agents unless there is a concrete architectural reason.
- Do not introduce LangGraph initially unless workflow complexity justifies stateful graphs, branching, retries, persistence, or human-in-the-loop workflows.
- LangChain is optional and must not be introduced merely because this is an AI project.

### RAG / vector search
- Do not introduce RAG or vector search unless a concrete retrieval problem justifies it.
- Live travel information should come from appropriate live APIs/providers rather than stale RAG data.

### Decision engine
- Must remain deterministic, testable, and explainable.
- Hard constraints must be enforced before recommendation scoring.
- Recommendations should be based on structured candidate data.
- Never fabricate unavailable metrics such as safety scores, prices, ratings, or availability.

### Tools / providers
- External services must be accessed through clear tool/provider interfaces.
- Providers should be replaceable.
- MVP may use mock providers.
- Mock data must never be represented as real-world live data.
- Never fake a successful booking.

### Approval / security
- Financial, booking, cancellation, or otherwise consequential actions require explicit user approval.
- Never expose secrets to the LLM.
- Validate tool inputs.
- Do not allow unrestricted arbitrary tool execution.
- Use environment variables for secrets.

## Working here

- Build incrementally.
- Prefer simple production-oriented architecture.
- Do not introduce microservices, Kubernetes, Kafka, or other infrastructure unless there is a genuine architectural requirement.
- Every meaningful feature should have tests.
- Do not modify unrelated files.
- Before implementing a substantial feature, inspect the existing repository and explain the implementation plan.
- Preserve existing architecture and interfaces unless a deliberate architectural change has been approved.
- Re-investigate the repo before relying on this file: tooling may have been added since this was written (run `git log` and look for manifests/config files).
- Keep `README.md` and `AGENTS.md` in sync with whatever scaffolding is introduced later.
