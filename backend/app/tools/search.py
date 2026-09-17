from __future__ import annotations

from decimal import Decimal
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from app.models.travel_requirements import TravelRequirements


class SearchRequest(BaseModel):
    """Handoff contract for the (future) travel search capability."""

    workflow_id: str
    requirements: TravelRequirements


class SearchResult(BaseModel):
    """One candidate found by a provider. Fields are present only when real data
    is available — never fabricated."""

    provider_ref: str
    label: str
    price: Decimal | None = None
    currency: str | None = None


class SearchResponse(BaseModel):
    workflow_id: str
    results: list[SearchResult] = Field(default_factory=list)


@runtime_checkable
class SearchProvider(Protocol):
    """External travel search boundary.

    M4 defines the contract only; no provider implementation exists yet.
    """

    def search(self, request: SearchRequest) -> SearchResponse: ...