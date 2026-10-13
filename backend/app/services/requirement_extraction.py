from __future__ import annotations

from datetime import datetime, timezone

from pydantic import ValidationError

from app.models.travel_requirements import (
    BudgetDetails,
    ClarificationImportance,
    ClarificationItem,
    CompanionInfo,
    ContextualSignals,
    DateRange,
    ExtractionMetadata,
    ExtractionResult,
    HardConstraints,
    SoftPreferences,
    TravelerDetails,
    TravelRequirements,
    TripDates,
    TripDetails,
)
from app.prompts.extraction import build_extraction_prompt
from app.providers.llm import (
    LLMClient,
    LLMClientError,
    LLMExtractionOutput,
    LLMExtractionRequest,
    BudgetExtraction,
    ContextualExtraction,
)


class ExtractionError(Exception):
    """Base error for the requirement extraction flow."""


class LLMOutputValidationError(ExtractionError):
    """Raised when the LLM structured output fails schema validation."""


class LLMProviderError(ExtractionError):
    """Raised when a live LLM provider fails (credentials, transport,
    rate limit, or an unreadable/unexpected response).

    ``code`` is a stable, credential-free error code surfaced to the agent.
    """

    def __init__(self, message: str, *, code: str = "llm_provider_error") -> None:
        super().__init__(message)
        self.code = code


class InconsistentExtractionError(ExtractionError):
    """Raised when the LLM output contains internally contradictory temporal data."""


def _reference_datetime() -> datetime:
    return datetime.now(timezone.utc)


def _reference_date_str(ref: datetime) -> str:
    return ref.date().isoformat()


class RequirementExtractionService:
    """Orchestrates LLM requirement extraction and deterministic transformation.

    Flow: user message (optionally with prior conversation context) -> LLMClient
    structured output -> definite Python transform -> ExtractionResult.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        reference_datetime: datetime | None = None,
    ) -> None:
        self._client = client
        self._reference_datetime = reference_datetime

    def extract(
        self,
        user_message: str,
        *,
        context: list[str] | None = None,
    ) -> ExtractionResult:
        """Extract requirements from ``user_message``.

        ``context`` carries prior user turns so a clarification answer can be
        reconciled against facts already stated. It is purely additive: when
        omitted, behaviour is identical to a single-message extraction.
        """
        reference_datetime = (
            self._reference_datetime
            if self._reference_datetime is not None
            else _reference_datetime()
        )
        system_prompt = build_extraction_prompt(
            reference_date=_reference_date_str(reference_datetime)
        )
        content = user_message
        if context:
            content = "\n".join([*context, user_message])
        request = LLMExtractionRequest(
            system_prompt=system_prompt,
            user_message=content,
            reference_datetime=reference_datetime,
        )
        try:
            output = self._client.generate_structured(
                request=request,
                schema=LLMExtractionOutput,
            )
        except ValidationError as exc:
            raise LLMOutputValidationError(
                f"LLM structured output failed validation: {exc}"
            ) from exc
        except LLMClientError as exc:
            raise LLMProviderError(str(exc), code=exc.code) from exc
        resulting_output = _as_output(output)
        return _to_extraction_result(resulting_output, original_message=user_message)


def _as_output(output: object) -> LLMExtractionOutput:
    if isinstance(output, LLMExtractionOutput):
        return output
    try:
        return LLMExtractionOutput.model_validate(output)
    except ValidationError as exc:
        raise LLMOutputValidationError(
            f"LLM structured output failed validation: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Deterministic transformation: LLMExtractionOutput -> ExtractionResult
# ---------------------------------------------------------------------------


def to_extraction_result(
    output: LLMExtractionOutput, *, original_message: str
) -> ExtractionResult:
    """Public entry point for the deterministic transformation.

    Exposed so a non-LLM (structured-input) adapter can reuse the *exact* same
    mapping and clarification policy as the LLM extraction path, rather than
    reimplementing it and risking drift.
    """
    return _to_extraction_result(output, original_message=original_message)


def _to_extraction_result(
    output: LLMExtractionOutput, *, original_message: str
) -> ExtractionResult:
    requirements = _build_requirements(output)
    clarification_needed = _build_clarifications(output)
    metadata = ExtractionMetadata(
        original_user_message=original_message,
        extraction_confidence=output.confidence,
        explicit_fields=_explicit_fields(output),
        inferred_fields=[],
        notes=list(output.notes),
    )
    return ExtractionResult(
        requirements=requirements,
        clarification_needed=clarification_needed,
        metadata=metadata,
    )


def _build_requirements(output: LLMExtractionOutput) -> TravelRequirements:
    dates: TripDates | None = None
    has_date_range = (
        output.trip_start_date is not None and output.trip_end_date is not None
    )
    has_duration = output.duration_days is not None
    if has_date_range:
        implied_days = (output.trip_end_date - output.trip_start_date).days
        if has_duration and implied_days != output.duration_days:
            raise InconsistentExtractionError(
                f"Date range {output.trip_start_date}..{output.trip_end_date} "
                f"implies {implied_days} day(s) but duration_days="
                f"{output.duration_days}"
            )
    if has_date_range or has_duration:
        date_range = (
            DateRange(start=output.trip_start_date, end=output.trip_end_date)
            if has_date_range
            else None
        )
        dates = TripDates(
            date_range=date_range,
            duration_days=output.duration_days,
            flexibility=(
                output.date_flexibility
                if output.date_flexibility is not None
                else TripDates().flexibility
            ),
        )

    trip = TripDetails(
        origin=output.trip_origin,
        destination=output.trip_destination,
        dates=dates,
    )

    travelers = TravelerDetails(
        count=output.traveler_count,
        categories=list(output.traveler_categories),
        companions=[
            CompanionInfo(
                relationship=c.relationship or "",
                categories=list(c.categories),
            )
            for c in output.companions
        ],
    )

    budget, hard_constraints = _build_budget(output.budget)

    soft_preferences = SoftPreferences(
        comfort_level=output.comfort_level,
        preferred_transport=output.preferred_transport,
        preferred_departure_time=output.departure_time,
    )
    if output.required_transport is not None:
        hard_constraints.required_transport = output.required_transport
    if output.arrival_time is not None:
        hard_constraints.arrival_time = output.arrival_time

    contextual = _build_contextual(output.contextual)

    return TravelRequirements(
        trip=trip,
        travelers=travelers,
        budget=budget,
        hard_constraints=hard_constraints,
        soft_preferences=soft_preferences,
        contextual_signals=contextual,
    )


def _build_budget(budget: BudgetExtraction | None) -> tuple[BudgetDetails, HardConstraints]:
    hard_constraints = HardConstraints()
    if budget is None:
        return BudgetDetails(), hard_constraints

    details = BudgetDetails(
        total=budget.amount,
        currency=budget.currency,
        is_approximate=(budget.kind.value == "approximate"),
    )

    if budget.kind.value == "hard_max" and budget.amount is not None:
        hard_constraints.max_budget = budget.amount

    return details, hard_constraints


def _build_contextual(contextual: ContextualExtraction | None) -> ContextualSignals:
    defaults = ContextualSignals()
    if contextual is None:
        return defaults
    return ContextualSignals(
        traveling_with_parents=(
            contextual.traveling_with_parents
            if contextual.traveling_with_parents is not None
            else defaults.traveling_with_parents
        ),
        has_children=(
            contextual.has_children
            if contextual.has_children is not None
            else defaults.has_children
        ),
        first_time_destination=(
            contextual.first_time_destination
            if contextual.first_time_destination is not None
            else defaults.first_time_destination
        ),
        accessibility_notes=contextual.accessibility_notes,
        dietary_notes=contextual.dietary_notes,
        other_notes=list(contextual.other_notes),
    )


def _explicit_fields(output: LLMExtractionOutput) -> list[str]:
    fields: list[str] = []
    if output.trip_origin is not None:
        fields.append("trip.origin")
    if output.trip_destination is not None:
        fields.append("trip.destination")
    if output.trip_start_date is not None and output.trip_end_date is not None:
        fields.append("trip.dates.date_range")
    if output.duration_days is not None:
        fields.append("trip.dates.duration_days")
    if output.traveler_count is not None:
        fields.append("travelers.count")
    if output.budget is not None and output.budget.amount is not None:
        fields.append("budget.total")
    return fields


# ---------------------------------------------------------------------------
# Clarification policy
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = {
    "trip.origin": "Where will you be travelling from?",
    "trip.destination": "Where would you like to go?",
    "travel.timing": "When would you like to travel, or for how many days?",
}

_RECOMMENDED_FIELDS = {
    "budget.total": "Do you have a budget in mind?",
    "travelers.count": "How many people are travelling?",
}


def _build_clarifications(output: LLMExtractionOutput) -> list[ClarificationItem]:
    items: list[ClarificationItem] = []

    required_missing: dict[str, str] = {}
    if output.trip_origin is None:
        required_missing["trip.origin"] = _REQUIRED_FIELDS["trip.origin"]
    if output.trip_destination is None:
        required_missing["trip.destination"] = _REQUIRED_FIELDS["trip.destination"]
    has_date_range = (
        output.trip_start_date is not None and output.trip_end_date is not None
    )
    has_duration = output.duration_days is not None
    timing_present = has_date_range or has_duration
    if not timing_present:
        required_missing["travel.timing"] = _REQUIRED_FIELDS["travel.timing"]

    for field, question in required_missing.items():
        items.append(
            ClarificationItem(
                field=field,
                importance=ClarificationImportance.REQUIRED,
                question=question,
            )
        )

    recommended_missing: dict[str, str] = {}
    if output.budget is None or output.budget.amount is None:
        recommended_missing["budget.total"] = _RECOMMENDED_FIELDS["budget.total"]
    if output.traveler_count is None:
        recommended_missing["travelers.count"] = _RECOMMENDED_FIELDS["travelers.count"]

    for field, question in recommended_missing.items():
        items.append(
            ClarificationItem(
                field=field,
                importance=ClarificationImportance.RECOMMENDED,
                question=question,
            )
        )

    covered_fields = {item.field for item in items}
    for cl in output.clarification_needed:
        if cl.field in covered_fields:
            item = next(it for it in items if it.field == cl.field)
            if cl.possible_interpretations:
                merged = [
                    p
                    for p in cl.possible_interpretations
                    if p not in item.possible_interpretations
                ]
                item.possible_interpretations.extend(merged)
            if cl.question and not item.question:
                item.question = cl.question
            continue
        items.append(
            ClarificationItem(
                field=cl.field,
                importance=(
                    cl.importance
                    if cl.importance is not None
                    else ClarificationImportance.RECOMMENDED
                ),
                question=cl.question or "",
                possible_interpretations=list(cl.possible_interpretations),
            )
        )

    return items
