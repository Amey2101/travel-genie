from __future__ import annotations

import json
import re
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
# Provider-level errors
# ---------------------------------------------------------------------------
#
# These describe failures *talking to* a provider (credentials, transport,
# rate limits, unreadable/unexpected payloads). They deliberately carry only
# fixed, credential-free messages. Domain/schema validation failures are NOT
# raised here: providers let ``pydantic.ValidationError`` propagate so the
# extraction service can keep deterministic validation authoritative.


class LLMClientError(Exception):
    """Base class for a safe, application-level LLM provider failure."""

    code = "llm_provider_error"


class LLMNotConfiguredError(LLMClientError):
    """The selected live provider has no credentials configured."""

    code = "llm_not_configured"


class LLMRequestError(LLMClientError):
    """Transport, authentication, rate-limit or provider HTTP failure."""

    code = "llm_request_failed"


class LLMResponseError(LLMClientError):
    """The provider responded, but the payload was missing or unreadable."""

    code = "llm_invalid_response"


# ---------------------------------------------------------------------------
# Shared structured-output helpers
# ---------------------------------------------------------------------------


def structured_system_prompt(system_prompt: str, schema: type[BaseModel]) -> str:
    """Append the target JSON Schema to a provider system instruction.

    Providers do not assume a native structured-output feature: the schema is
    supplied in-band, then the returned JSON is validated against it.
    """
    schema_json = json.dumps(
        schema.model_json_schema(), indent=2, ensure_ascii=False
    )
    return (
        f"{system_prompt}\n\n"
        "Respond with a single JSON object that conforms to the JSON Schema "
        "below. Return raw JSON only: no markdown code fences and no "
        "commentary. Omit unknown fields rather than guessing values.\n"
        f"JSON Schema:\n{schema_json}"
    )


_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from model output, tolerating fences/prose.

    Raises ``LLMResponseError`` (never a raw provider exception) when the text
    is empty, malformed, or not a JSON object.
    """
    candidate = (text or "").strip()
    if not candidate:
        raise LLMResponseError("The provider returned an empty response.")
    candidate = _CODE_FENCE.sub("", candidate)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise LLMResponseError("The provider returned malformed JSON.")
        try:
            data = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMResponseError("The provider returned malformed JSON.") from exc
    if not isinstance(data, dict):
        raise LLMResponseError(
            "The provider returned JSON in an unexpected shape."
        )
    return data


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
