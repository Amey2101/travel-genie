"""Deterministic structured-input adapter for the requirement pipeline.

The home-page form collects discrete fields, while the conversational workflow
takes free-text messages. Rather than concatenate fields into a sentence and
hope an LLM interprets them (the default ``stub`` provider interprets nothing),
this adapter deterministically maps validated form fields into the very same
``LLMExtractionOutput`` shape the LLM boundary produces, then reuses
``to_extraction_result`` so clarification policy, budget semantics and state
transitions are identical to the LLM path.

Invariants preserved:
- The stated budget is a *soft target* (``approximate``); it is never silently
  promoted to a hard maximum.
- Trip dates/duration are validated; inconsistent inputs are rejected by the
  request model (HTTP 422) before any workflow state is touched.
- No search, decision or orchestration logic lives here: the result is handed
  back to ``TravelAgent``, which remains the single orchestrator.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.travel_requirements import (
    BudgetDetails,
    ComfortLevel,
    ExtractionResult,
    TravelRequirements,
)
from app.providers.llm import (
    BudgetExtraction,
    BudgetKind,
    ContextualExtraction,
    LLMExtractionOutput,
)
from app.services.requirement_extraction import to_extraction_result


class ClarificationError(Exception):
    """A deterministic clarification answer could not be applied safely."""


SUPPORTED_CURRENCIES = ("INR", "USD", "EUR", "GBP", "AED", "SGD", "THB")
KNOWN_INTERESTS = (
    "food",
    "nature",
    "adventure",
    "history",
    "beaches",
    "shopping",
    "culture",
    "nightlife",
)
TRAVEL_STYLES: dict[str, ComfortLevel] = {
    "budget": ComfortLevel.BUDGET,
    "balanced": ComfortLevel.STANDARD,
    "luxury": ComfortLevel.LUXURY,
}


class StructuredRequirementsRequest(BaseModel):
    """Validated structured trip request submitted by the home-page form."""

    origin: str = Field(min_length=1, max_length=80)
    destination: str = Field(min_length=1, max_length=80)
    departure_date: date
    return_date: date | None = None
    duration_days: int | None = Field(default=None, ge=1, le=60)
    travelers: int = Field(default=1, ge=1, le=20)
    budget: Decimal | None = Field(default=None, gt=0)
    currency: str = "INR"
    travel_style: Literal["budget", "balanced", "luxury"] = "balanced"
    interests: list[str] = Field(default_factory=list)
    accommodation: str = Field(default="", max_length=100)
    notes: str = Field(default="", max_length=1000)

    @field_validator("origin", "destination", "accommodation", "notes", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("currency")
    @classmethod
    def _currency(cls, value: str) -> str:
        normalised = value.strip().upper()
        if normalised not in SUPPORTED_CURRENCIES:
            raise ValueError(
                f"Currency must be one of: {', '.join(SUPPORTED_CURRENCIES)}."
            )
        return normalised

    @field_validator("interests")
    @classmethod
    def _interests(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in values:
            normalised = str(item).strip().lower()
            if normalised not in KNOWN_INTERESTS:
                raise ValueError(
                    f"Unknown interest {normalised!r}. "
                    f"Choose from: {', '.join(KNOWN_INTERESTS)}."
                )
            if normalised not in cleaned:
                cleaned.append(normalised)
        return cleaned

    @model_validator(mode="after")
    def _dates(self) -> "StructuredRequirementsRequest":
        if self.return_date is not None:
            if self.return_date < self.departure_date:
                raise ValueError("Return date must be on or after the departure date.")
            if self.duration_days is not None:
                implied = (self.return_date - self.departure_date).days
                if implied != self.duration_days:
                    raise ValueError(
                        "Return date and duration do not match. "
                        "Provide only one, or make them agree."
                    )
        return self


def _summary(request: StructuredRequirementsRequest) -> str:
    parts = [f"{request.origin} to {request.destination}"]
    if request.return_date is not None:
        parts.append(
            f"{request.departure_date.isoformat()} to {request.return_date.isoformat()}"
        )
    else:
        parts.append(request.departure_date.isoformat())
        if request.duration_days is not None:
            parts.append(f"{request.duration_days} day(s)")
    parts.append(f"{request.travelers} traveler(s)")
    if request.budget is not None:
        parts.append(f"budget {request.budget} {request.currency}".rstrip())
    return ", ".join(parts)


def build_extraction_result(
    request: StructuredRequirementsRequest,
) -> ExtractionResult:
    """Map validated form fields into the deterministic extraction pipeline.

    Reuses the LLM path's transform so behaviour stays consistent. The budget is
    always ``approximate`` here: the form asks for a target, not a ceiling.
    """
    start = request.departure_date
    end = request.return_date
    duration = None if end is not None else request.duration_days

    budget = (
        BudgetExtraction(
            amount=request.budget,
            currency=request.currency,
            kind=BudgetKind.APPROXIMATE,
        )
        if request.budget is not None
        else None
    )

    output = LLMExtractionOutput(
        trip_origin=request.origin,
        trip_destination=request.destination,
        trip_start_date=start,
        trip_end_date=end,
        duration_days=duration,
        traveler_count=request.travelers,
        budget=budget,
        comfort_level=TRAVEL_STYLES[request.travel_style],
    )
    result = to_extraction_result(output, original_message=_summary(request))

    result.requirements.soft_preferences.priorities = list(request.interests)
    notes: list[str] = []
    if request.accommodation:
        notes.append(f"Accommodation preference: {request.accommodation}")
    if request.notes:
        notes.append(request.notes)
    result.requirements.contextual_signals.other_notes = notes
    return result


class ClarificationAnswer(BaseModel):
    """A deterministic answer to open clarification questions.

    Every field is optional: callers provide only what was missing (or what they
    want to correct). The answer is merged into the existing, already-extracted
    requirements and re-runs the shared clarification/search/decision pipeline —
    no LLM is involved, so this works with the credential-free ``stub`` provider.
    """

    origin: str | None = Field(default=None, max_length=80)
    destination: str | None = Field(default=None, max_length=80)
    departure_date: date | None = None
    return_date: date | None = None
    duration_days: int | None = Field(default=None, ge=1, le=60)
    travelers: int | None = Field(default=None, ge=1, le=20)
    budget: Decimal | None = Field(default=None, gt=0)
    currency: str | None = None

    @field_validator("origin", "destination", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("currency")
    @classmethod
    def _currency(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalised = value.strip().upper()
        if normalised not in SUPPORTED_CURRENCIES:
            raise ValueError(
                f"Currency must be one of: {', '.join(SUPPORTED_CURRENCIES)}."
            )
        return normalised

    @model_validator(mode="after")
    def _timing(self) -> "ClarificationAnswer":
        if self.return_date is not None and self.duration_days is not None:
            raise ValueError("Provide either a return date or a duration, not both.")
        if (
            self.return_date is not None
            and self.departure_date is not None
            and self.return_date < self.departure_date
        ):
            raise ValueError("Return date must be on or after the departure date.")
        return self


def _carry_over(base: TravelRequirements) -> LLMExtractionOutput:
    """Rebuild the LLM-boundary shape from existing requirements.

    Used so a clarification answer can be merged without re-running the LLM and
    without a second mapping path: everything not supplied by the answer is
    carried over verbatim from the current requirements.
    """
    dates = base.trip.dates
    date_range = dates.date_range if dates is not None else None
    return LLMExtractionOutput(
        trip_origin=base.trip.origin,
        trip_destination=base.trip.destination,
        trip_start_date=date_range.start if date_range is not None else None,
        trip_end_date=date_range.end if date_range is not None else None,
        duration_days=(
            dates.duration_days
            if dates is not None and date_range is None
            else None
        ),
        date_flexibility=dates.flexibility if dates is not None else None,
        traveler_count=base.travelers.count,
        comfort_level=base.soft_preferences.comfort_level,
        preferred_transport=base.soft_preferences.preferred_transport,
        departure_time=base.soft_preferences.preferred_departure_time,
        required_transport=base.hard_constraints.required_transport,
        arrival_time=base.hard_constraints.arrival_time,
        contextual=ContextualExtraction(
            traveling_with_parents=base.contextual_signals.traveling_with_parents,
            has_children=base.contextual_signals.has_children,
            first_time_destination=base.contextual_signals.first_time_destination,
            accessibility_notes=base.contextual_signals.accessibility_notes,
            dietary_notes=base.contextual_signals.dietary_notes,
            other_notes=list(base.contextual_signals.other_notes),
        ),
    )


def _clarification_summary(answer: ClarificationAnswer) -> str:
    parts: list[str] = []
    if answer.origin:
        parts.append(f"from {answer.origin}")
    if answer.destination:
        parts.append(f"to {answer.destination}")
    if answer.departure_date is not None:
        parts.append(f"departing {answer.departure_date.isoformat()}")
    if answer.return_date is not None:
        parts.append(f"returning {answer.return_date.isoformat()}")
    if answer.duration_days is not None:
        parts.append(f"for {answer.duration_days} day(s)")
    if answer.travelers is not None:
        parts.append(f"{answer.travelers} traveler(s)")
    if answer.budget is not None:
        parts.append(f"budget {answer.budget} {answer.currency or ''}".rstrip())
    return "Clarification answer: " + (", ".join(parts) if parts else "(no changes)")


def merge_clarification(
    base: ExtractionResult, answer: ClarificationAnswer
) -> ExtractionResult:
    """Apply a deterministic ``ClarificationAnswer`` to existing requirements.

    Reuses ``to_extraction_result`` so the clarification policy, budget
    semantics and date invariants stay identical to every other path. Only
    fields explicitly supplied are changed; everything else is preserved.
    """
    requirements = base.requirements
    output = _carry_over(requirements)

    if answer.origin is not None:
        output.trip_origin = answer.origin
    if answer.destination is not None:
        output.trip_destination = answer.destination
    if answer.travelers is not None:
        output.traveler_count = answer.travelers

    start = output.trip_start_date
    end = output.trip_end_date
    duration = output.duration_days
    if answer.departure_date is not None:
        start = answer.departure_date
    if answer.return_date is not None:
        end = answer.return_date
        duration = None
    elif answer.duration_days is not None:
        duration = answer.duration_days
        end = None
    if end is not None:
        if start is None:
            raise ClarificationError(
                "A departure date is required when a return date is provided."
            )
        if end < start:
            raise ClarificationError(
                "Return date must be on or after the departure date."
            )
        duration = None
    output.trip_start_date = start
    output.trip_end_date = end
    output.duration_days = duration

    result = to_extraction_result(
        output, original_message=_clarification_summary(answer)
    )
    result.requirements.soft_preferences.priorities = list(
        requirements.soft_preferences.priorities
    )
    if answer.budget is not None:
        result.requirements.budget = BudgetDetails(
            total=answer.budget,
            currency=answer.currency or requirements.budget.currency,
            is_approximate=True,
        )
    else:
        result.requirements.budget = requirements.budget
    result.requirements.hard_constraints.max_budget = (
        requirements.hard_constraints.max_budget
    )
    return result
