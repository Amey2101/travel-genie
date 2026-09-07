from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TravelerCategory(str, Enum):
    """Age/life-stage category — only populated when the user explicitly states it."""

    ADULT = "adult"
    CHILD = "child"
    INFANT = "infant"
    SENIOR = "senior"


class ComfortLevel(str, Enum):
    BUDGET = "budget"
    ECONOMY = "economy"
    STANDARD = "standard"
    PREMIUM = "premium"
    LUXURY = "luxury"


class TransportMode(str, Enum):
    FLIGHT = "flight"
    TRAIN = "train"
    BUS = "bus"
    CAR = "car"
    FERRY = "ferry"
    ANY = "any"


class TimePreference(str, Enum):
    EARLY_MORNING = "early_morning"
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NO_PREFERENCE = "no_preference"


class DateFlexibility(str, Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    FLEXIBLE = "flexible"


class ClarificationImportance(str, Enum):
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


# ---------------------------------------------------------------------------
# Trip
# ---------------------------------------------------------------------------


class DateRange(BaseModel):
    start: date
    end: date


class TripDates(BaseModel):
    """Flexible date representation.

    Both ``date_range`` and ``duration_days`` may be absent when the user
    has not provided any timing information.  Missing timing is surfaced
    through ``clarification_needed`` in ``ExtractionResult``.
    """

    date_range: DateRange | None = None
    duration_days: int | None = None
    flexibility: DateFlexibility = DateFlexibility.EXACT


class TripDetails(BaseModel):
    origin: str | None = None
    destination: str | None = None
    dates: TripDates | None = None


# ---------------------------------------------------------------------------
# Travelers
# ---------------------------------------------------------------------------


class CompanionInfo(BaseModel):
    """Contextual information about people the user is travelling with.

    ``relationship`` is stored verbatim from the user's words.
    ``categories`` is populated *only* when the user explicitly states age
    groups — it must not be inferred.
    """

    relationship: str
    categories: list[TravelerCategory] = Field(default_factory=list)
    notes: str | None = None


class TravelerDetails(BaseModel):
    """Composition of the travelling party.

    ``count`` is the total number of people on the trip, including the
    speaker.  When the user says "I'm taking my parents", count = 3
    (speaker + two parents) *if* the message clearly includes the speaker
    in the trip.  When the message does not support a count, count is None
    and a clarification item is added.

    ``categories`` is populated *only* for explicitly stated age groups.
    ``companions`` captures relational context (e.g. "with parents")
    without imposing unsupported assumptions.
    """

    count: int | None = Field(default=None, ge=1)
    categories: list[TravelerCategory] = Field(default_factory=list)
    companions: list[CompanionInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class BudgetDetails(BaseModel):
    """What the user stated about their budget.

    This is **informational only** — it is never enforced as a constraint.
    A user saying "around ₹30,000" populates this model.
    A user saying "do not exceed ₹30,000" populates this model **and**
    ``HardConstraints.max_budget``.
    """

    total: Decimal | None = None
    currency: str | None = None
    is_approximate: bool = False
    per_category: dict[str, Decimal] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hard constraints / Soft preferences / Contextual signals
# ---------------------------------------------------------------------------


class HardConstraints(BaseModel):
    """Enforceable requirements.  A candidate must satisfy every non-None
    field or it is rejected.

    Only populated from explicit, concrete, enforceable user statements.
    "Around ₹30k" does **not** set ``max_budget``.
    "Do not exceed ₹30k" **does**.
    Destination is not duplicated here — it lives in ``TripDetails``.
    """

    max_budget: Decimal | None = None
    min_duration_days: int | None = None
    max_duration_days: int | None = None
    date_range: DateRange | None = None
    departure_time: TimePreference | None = None
    arrival_time: TimePreference | None = None
    required_transport: TransportMode | None = None


class SoftPreferences(BaseModel):
    """Ranking signals used to score and order candidates, never to reject."""

    comfort_level: ComfortLevel | None = None
    preferred_transport: TransportMode | None = None
    preferred_departure_time: TimePreference | None = None
    hotel_rating_min: float | None = None
    hotel_proximity: str | None = None
    priorities: list[str] = Field(default_factory=list)


class ContextualSignals(BaseModel):
    """Informational context that may influence planning or trigger
    clarification but is **never** automatically enforced as a constraint.

    "Travelling with parents" is stored here — it is never converted to
    assumptions about age, mobility, or accessibility.
    """

    traveling_with_parents: bool = False
    has_children: bool = False
    first_time_destination: bool = False
    accessibility_notes: str | None = None
    dietary_notes: str | None = None
    other_notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


class TravelRequirements(BaseModel):
    """Pure domain object representing what a user wants from a trip.

    Contains no extraction metadata, no provenance tracking, and no
    ambiguity information — those belong in ``ExtractionResult``.
    """

    trip: TripDetails
    travelers: TravelerDetails
    budget: BudgetDetails
    hard_constraints: HardConstraints
    soft_preferences: SoftPreferences
    contextual_signals: ContextualSignals


# ---------------------------------------------------------------------------
# Extraction pipeline types
# ---------------------------------------------------------------------------


class ClarificationItem(BaseModel):
    """A single piece of missing or ambiguous information."""

    field: str
    importance: ClarificationImportance
    question: str
    current_value: str | None = None
    possible_interpretations: list[str] = Field(default_factory=list)


class ExtractionMetadata(BaseModel):
    """Provenance and confidence for the extraction step.

    ``extraction_confidence`` is a raw score — no fixed thresholds are
    applied.  Thresholds will be determined empirically later.  It is
    ``None`` when the LLM provided no confidence value; confidence is
    metadata only and never drives behavior.
    """

    original_user_message: str
    extraction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    explicit_fields: list[str] = Field(default_factory=list)
    inferred_fields: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Pipeline artifact wrapping the domain model with extraction context.

    The extraction layer produces this.  The clarification / prompting
    system reads ``clarification_needed``.  The decision engine receives
    only the inner ``TravelRequirements``.
    """

    requirements: TravelRequirements
    clarification_needed: list[ClarificationItem] = Field(default_factory=list)
    metadata: ExtractionMetadata
