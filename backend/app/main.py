from fastapi import FastAPI

from app.agent.orchestrator import TravelAgent
from app.agent.store import InMemoryWorkflowStore
from app.api.health import router as health_router
from app.api.workflows import workflows_router
from app.providers.llm import LLMExtractionOutput
from app.providers.stub_llm import StubLLMClient
from app.services.requirement_extraction import RequirementExtractionService


def _default_agent() -> TravelAgent:
    stub = StubLLMClient(LLMExtractionOutput())
    extraction = RequirementExtractionService(stub)
    return TravelAgent(extraction=extraction, store=InMemoryWorkflowStore())


def create_app(agent: TravelAgent | None = None) -> FastAPI:
    if agent is None:
        agent = _default_agent()
    app = FastAPI(title="Travel Genie", version="0.2.0")
    app.include_router(health_router)
    app.include_router(workflows_router(agent))
    return app


app = create_app()
