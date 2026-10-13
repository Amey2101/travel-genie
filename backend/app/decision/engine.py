from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from app.models.travel_requirements import (
    TimePreference,
    TransportMode,
    TravelRequirements,
)
from app.tools.search import ResultSource, SearchResult


class RecommendationStatus(str, Enum):
    """Explicit outcome of a planning/decision attempt.

    ``PROVIDER_UNAVAILABLE`` and ``DECISION_ERROR`` are workflow-level outcomes
    recorded by the orchestrator; the engine itself returns ``OK``,
    ``NO_RESULTS`` or ``NO_FEASIBLE_OPTIONS``.
    """

    OK = "ok"
    NO_RESULTS = "no_results"
    NO_FEASIBLE_OPTIONS = "no_feasible_options"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    DECISION_ERROR = "decision_error"


class Candidate(BaseModel):
    """A structured candidate to be scored.

    Metrics are present only when real — never fabricated. ``source`` records
    provenance (verified / estimate / demo) and ``bookable`` is ``False`` for
    anything that is not a confirmed purchasable product.
    """

    provider_ref: str
    label: str
    price: Decimal | None = None
    currency: str | None = None
    source: ResultSource = ResultSource.DEMO
    bookable: bool = False
    category: str | None = None
    transport_mode: TransportMode | None = None
    duration_days: int | None = None
    departure_time: TimePreference | None = None
    arrival_time: TimePreference | None = None
    start_date: date | None = None
    end_date: date | None = None

    @classmethod
    def from_search_result(cls, result: SearchResult) -> "Candidate":
        """Adapt a provider ``SearchResult`` into a scoreable candidate."""
        return cls(**result.model_dump())


class RecommendationInput(BaseModel):
    workflow_id: str
    requirements: TravelRequirements
    candidates: list[Candidate] = Field(default_factory=list)


class RejectedCandidate(BaseModel):
    """A candidate that failed at least one hard constraint."""

    candidate: Candidate
    reasons: list[str] = Field(default_factory=list)


class RecommendedOption(BaseModel):
    rank: int = Field(ge=1)
    candidate: Candidate
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    workflow_id: str
    status: RecommendationStatus = RecommendationStatus.OK
    options: list[RecommendedOption] = Field(default_factory=list)
    rejected: list[RejectedCandidate] = Field(default_factory=list)


@runtime_checkable
class DecisionEngine(Protocol):
    """Deterministic decision boundary.

    Contract: hard constraints must be enforced (filtering) before any scoring,
    unavailable metrics must never be fabricated, and the engine must not call
    an LLM or make approval/execution decisions.
    """

    def recommend(self, input_: RecommendationInput) -> Recommendation: ...
