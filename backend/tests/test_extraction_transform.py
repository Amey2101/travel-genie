from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Callable

from app.models.travel_requirements import (
    BudgetDetails,
    DateFlexibility,
    DateRange,
    TransportMode,
    TravelerCategory,
    TripDates,
)
from app.providers.llm import (
    BudgetExtraction,
    BudgetKind,
    CompanionExtraction,
    ContextualExtraction,
    LLMExtractionOutput,
)
from app.services.requirement_extraction import RequirementExtractionService

MakeService = Callable[[LLMExtractionOutput], RequirementExtractionService]


def _full_output() -> LLMExtractionOutput:
    return LLMExtractionOutput(
        trip_origin="Mumbai",
        trip_destination="Goa",
        trip_start_date=date(2026, 12, 20),
        trip_end_date=date(2026, 12, 24),
        duration_days=4,
        date_flexibility=DateFlexibility.APPROXIMATE,
        traveler_count=4,
        traveler_categories=[
            TravelerCategory.ADULT,
            TravelerCategory.ADULT,
            TravelerCategory.CHILD,
            TravelerCategory.CHILD,
        ],
        companions=[CompanionExtraction(relationship="family")],
        budget=BudgetExtraction(amount=Decimal("50000"), currency="INR", kind=BudgetKind.APPROXIMATE),
        required_transport=TransportMode.FLIGHT,
        preferred_transport=TransportMode.FLIGHT,
        contextual=ContextualExtraction(
            traveling_with_parents=False,
            has_children=True,
        ),
        confidence=0.93,
        notes=["user prefers window seats"],
    )


class TestTransformation:
    def test_full_output_maps_to_requirements(self, make_service: MakeService) -> None:
        service = make_service(_full_output())
        result = service.extract("whatever")
        req = result.requirements

        assert req.trip.origin == "Mumbai"
        assert req.trip.destination == "Goa"
        assert isinstance(req.trip.dates, TripDates)
        assert req.trip.dates.date_range == DateRange(
            start=date(2026, 12, 20), end=date(2026, 12, 24)
        )
        assert req.trip.dates.duration_days == 4
        assert req.trip.dates.flexibility == DateFlexibility.APPROXIMATE
        assert req.travelers.count == 4
        assert req.travelers.categories == [
            TravelerCategory.ADULT,
            TravelerCategory.ADULT,
            TravelerCategory.CHILD,
            TravelerCategory.CHILD,
        ]
        assert req.travelers.companions[0].relationship == "family"
        assert req.hard_constraints.required_transport == TransportMode.FLIGHT
        assert req.soft_preferences.preferred_transport == TransportMode.FLIGHT
        assert req.contextual_signals.has_children is True
        assert req.contextual_signals.traveling_with_parents is False

    def test_metadata_and_clarifications(self, make_service: MakeService) -> None:
        service = make_service(_full_output())
        result = service.extract("whatever")

        assert result.metadata.extraction_confidence == 0.93
        assert result.metadata.notes == ["user prefers window seats"]
        assert result.metadata.explicit_fields == [
            "trip.origin",
            "trip.destination",
            "trip.dates.date_range",
            "trip.dates.duration_days",
            "travelers.count",
            "budget.total",
        ]
        assert result.clarification_needed == []

    def test_minimal_output_no_explicit_dates(self, make_service: MakeService) -> None:
        service = make_service(LLMExtractionOutput(trip_destination="Goa"))
        result = service.extract("Goa trip")

        assert result.requirements.trip.destination == "Goa"
        assert result.requirements.trip.dates is None
        assert result.requirements.budget == BudgetDetails()

    def test_dates_without_range_only_duration(self, make_service: MakeService) -> None:
        service = make_service(
            LLMExtractionOutput(trip_destination="Goa", duration_days=3)
        )
        result = service.extract("Goa, 3 days")
        req = result.requirements
        assert req.trip.dates is not None
        assert req.trip.dates.date_range is None
        assert req.trip.dates.duration_days == 3

    def test_concrete_range_and_flexibility_preserved_together(
        self, make_service: MakeService
    ) -> None:
        output = LLMExtractionOutput(
            trip_destination="Goa",
            trip_start_date=date(2026, 12, 20),
            trip_end_date=date(2026, 12, 24),
            date_flexibility=DateFlexibility.FLEXIBLE,
        )
        req = make_service(output).extract("Goa around Dec 20-24, flexible").requirements
        assert req.trip.dates.date_range == DateRange(
            start=date(2026, 12, 20), end=date(2026, 12, 24)
        )
        assert req.trip.dates.flexibility == DateFlexibility.FLEXIBLE