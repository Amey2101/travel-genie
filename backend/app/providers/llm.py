from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, Field

from app.models.travel_requirements import (
    ClarificationImportance,
    ComfortLevel,
    DateFlexibility,
    TimePreference,
    TransportMode,
    TravelerCategory,
)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# LLM boundary
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """Replaceable boundary for any structured-output LLM provider.

    The extraction service depends only on this abstraction and never on a
    concrete provider (e.g. OpenRouter).
    """

    def generate_structured(self, *, request: LLMExtractionRequest, schema: type[T]) -> T: ...


class LLMExtractionRequest(BaseModel):
    """Everything the LLM needs to produce a structured extraction."""

    system_prompt: str
    user_message: str
    reference_datetime: datetime


# ---------------------------------------------------------------------------
# LLM-side extraction output (typed structured schema)
# ---------------------------------------------------------------------------


class BudgetKind(str, Enum):
    APPROXIMATE = "approximate"
    HARD_MAX = "hard_max"
    UNSPECIFIED = "unspecified"


class BudgetExtraction(BaseModel):
    amount: Decimal | None = None
    currency: str | None = None
    kind: BudgetKind = BudgetKind.UNSPECIFIED


class CompanionExtraction(BaseModel):
    relationship: str | None = None
    categories: list[TravelerCategory] = Field(default_factory=list)


class ContextualExtraction(BaseModel):
    traveling_with_parents: bool | None = None
    has_children: bool | None = None
    first_time_destination: bool | None = None
    accessibility_notes: str | None = None
    dietary_notes: str | None = None
    other_notes: list[str] = Field(default_factory=list)


class ClarificationExtraction(BaseModel):
    field: str
    importance: ClarificationImportance | None = None
    question: str | None = None
    possible_interpretations: list[str] = Field(default_factory=list)


class LLMExtractionOutput(BaseModel):
    """Typed structured output the LLM is asked to produce.

    This is a flat, fact-level representation — deterministic Python is
    responsible for mapping it into the ``TravelRequirements`` domain model.
    """

    trip_origin: str | None = None
    trip_destination: str | None = None
    trip_start_date: date | None = None
    trip_end_date: date | None = None
    duration_days: int | None = Field(default=None, ge=1)
    date_flexibility: DateFlexibility | None = None
    traveler_count: int | None = Field(default=None, ge=1)
    traveler_categories: list[TravelerCategory] = Field(default_factory=list)
    companions: list[CompanionExtraction] = Field(default_factory=list)
    budget: BudgetExtraction | None = None
    required_transport: TransportMode | None = None
    preferred_transport: TransportMode | None = None
    comfort_level: ComfortLevel | None = None
    departure_time: TimePreference | None = None
    arrival_time: TimePreference | None = None
    contextual: ContextualExtraction | None = None
    clarification_needed: list[ClarificationExtraction] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)
