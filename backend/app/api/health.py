from fastapi import APIRouter
from pydantic import BaseModel

from app.core import config

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str


class IntegrationInfo(BaseModel):
    configured: bool
    model: str | None = None


class Integrations(BaseModel):
    gemini: IntegrationInfo
    groq: IntegrationInfo
    serpapi: IntegrationInfo


class ApiHealthResponse(BaseModel):
    """Setup summary. Reports whether keys are configured, never their values."""

    status: str
    mode: str
    env_file_found: bool
    integrations: Integrations


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/api/health", response_model=ApiHealthResponse)
def api_health() -> ApiHealthResponse:
    gemini_ok = bool(config.gemini_key())
    groq_ok = bool(config.groq_key())
    return ApiHealthResponse(
        status="ok",
        mode="live" if gemini_ok or groq_ok else "demo",
        env_file_found=config.env_file_found(),
        integrations=Integrations(
            gemini=IntegrationInfo(configured=gemini_ok, model=config.gemini_model()),
            groq=IntegrationInfo(configured=groq_ok, model=config.groq_model()),
            serpapi=IntegrationInfo(configured=bool(config.serpapi_key())),
        ),
    )
