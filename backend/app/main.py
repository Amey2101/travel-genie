import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent.orchestrator import TravelAgent
from app.agent.store import InMemoryWorkflowStore
from app.api.discover import discover_router
from app.api.errors import register_exception_handlers
from app.api.health import router as health_router
from app.api.plan import router as plan_router
from app.api.workflows import workflows_router
from app.core import config
from app.decision.budget import StyleBasedBudgetEstimator
from app.decision.deterministic import DeterministicDecisionEngine
from app.providers.factory import build_llm_client, build_search_provider
from app.services.requirement_extraction import RequirementExtractionService
from app.tools.discovery import DiscoveryProvider

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

log = logging.getLogger("travel_genie.app")


def setup_logging() -> None:
    """Configure the ``travel_genie`` logger (uvicorn only configures its own).

    Verbosity comes from ``LOG_LEVEL`` (default INFO). Secrets are never logged.
    """
    logger = logging.getLogger("travel_genie")
    logger.setLevel(getattr(logging, config.log_level(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        logger.addHandler(handler)


setup_logging()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Booleans only; key values are never logged.
    log.info(
        "Startup: env_file_found=%s GEMINI_API_KEY_set=%s GROQ_API_KEY_set=%s "
        "SERPAPI_API_KEY_set=%s",
        config.env_file_found(),
        bool(config.gemini_key()),
        bool(config.groq_key()),
        bool(config.serpapi_key()),
    )
    yield


def _default_agent() -> TravelAgent:
    extraction = RequirementExtractionService(build_llm_client())
    return TravelAgent(
        extraction=extraction,
        store=InMemoryWorkflowStore(),
        search_provider=build_search_provider(),
        decision_engine=DeterministicDecisionEngine(),
        budget_estimator=StyleBasedBudgetEstimator(),
    )


def create_app(
    agent: TravelAgent | None = None,
    discovery_provider: DiscoveryProvider | None = None,
) -> FastAPI:
    if agent is None:
        agent = _default_agent()
    app = FastAPI(title="Travel Genie", version="0.4.0", lifespan=lifespan)
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(workflows_router(agent))
    app.include_router(plan_router)
    app.include_router(discover_router(discovery_provider))

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/discover", include_in_schema=False)
    def discover_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "discover.html")

    return app


app = create_app()
