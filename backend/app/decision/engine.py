from __future__ import annotations

from decimal import Decimal
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from app.models.travel_requirements import TravelRequirements


class Candidate(BaseModel):
    """A structured candidate to be scored.

    Metrics are present only when real — never fabricated.
    """

    provider_ref: str
    label: str
    price: Decimal | None = None
    currency: str | None = None


class RecommendationInput(BaseModel):
    workflow_id: str
    requirements: TravelRequirements
    candidates: list[Candidate] = Field(default_factory=list)


class RecommendedOption(BaseModel):
    rank: int = Field(ge=1)
    candidate: Candidate
    reasons: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    workflow_id: str
    options: list[RecommendedOption] = Field(default_factory=list)


@runtime_checkable
class DecisionEngine(Protocol):
    """Deterministic decision boundary.

    Contract: hard constraints must be enforced (filtering) before any scoring,
    and unavailable metrics must never be fabricated.
    """

    def recommend(self, input_: RecommendationInput) -> Recommendation: ...