from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.models.travel_requirements import (
    BudgetDetails,
    ClarificationImportance,
    ClarificationItem,
    ComfortLevel,
    ContextualSignals,
    DateFlexibility,
    DateRange,
    ExtractionMetadata,
    ExtractionResult,
    HardConstraints,
    SoftPreferences,
    TimePreference,
    TravelerCategory,
    TravelerDetails,
    TravelRequirements,
    TransportMode,
    TripDates,
    TripDetails,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _complete_requirements() -> TravelRequirements:
    """A fully populated, valid TravelRequirements."""
    return TravelRequirements(
        trip=TripDetails(
            origin="Mumbai",
            destination="Goa",
            dates=TripDates(
                date_range=DateRange(start=date(2025, 12, 20), end=date(2025, 12, 24)),
                flexibility=DateFlexibility.EXACT,
            ),
        ),
        travelers=TravelerDetails(
            count=3,
            categories=[TravelerCategory.ADULT],
            companions=[],
        ),
        budget=BudgetDetails(
            total=Decimal("50000"),
            currency="INR",
            is_approximate=False,
        ),
        hard_constraints=HardConstraints(
            max_budget=Decimal("50000"),
            max_duration_days=5,
        ),
        soft_preferences=SoftPreferences(
            comfort_level=ComfortLevel.STANDARD,
            preferred_transport=TransportMode.FLIGHT,
            hotel_proximity="near beach",
        ),
        contextual_signals=ContextualSignals(
            traveling_with_parents=False,
            has_children=False,
        ),
    )


# ---------------------------------------------------------------------------
# 1. Complete valid TravelRequirements
# ---------------------------------------------------------------------------


class TestCompleteTravelRequirements:
    def test_round_trip_serialisation(self) -> None:
        req = _complete_requirements()
        data = req.model_dump()
        restored = TravelRequirements.model_validate(data)
        assert restored.trip.destination == "Goa"
        assert restored.travelers.count == 3
        assert restored.budget.total == Decimal("50000")
        assert restored.hard_constraints.max_budget == Decimal("50000")
        assert restored.soft_preferences.comfort_level == ComfortLevel.STANDARD

    def test_json_round_trip(self) -> None:
        req = _complete_requirements()
        json_str = req.model_dump_json()
        restored = TravelRequirements.model_validate_json(json_str)
        assert restored.trip.origin == "Mumbai"


# ---------------------------------------------------------------------------
# 2. Incomplete trip — missing dates
# ---------------------------------------------------------------------------


class TestIncompleteTrip:
    def test_missing_dates(self) -> None:
        req = TravelRequirements(
            trip=TripDetails(
                origin="Mumbai",
                destination="Goa",
                dates=None,
            ),
            travelers=TravelerDetails(count=2),
            budget=BudgetDetails(),
            hard_constraints=HardConstraints(),
            soft_preferences=SoftPreferences(),
            contextual_signals=ContextualSignals(),
        )
        assert req.trip.dates is None

    def test_dates_present_but_no_range_or_duration(self) -> None:
        req = TravelRequirements(
            trip=TripDetails(
                destination="Goa",
                dates=TripDates(flexibility=DateFlexibility.FLEXIBLE),
            ),
            travelers=TravelerDetails(count=1),
            budget=BudgetDetails(),
            hard_constraints=HardConstraints(),
            soft_preferences=SoftPreferences(),
            contextual_signals=ContextualSignals(),
        )
        assert req.trip.dates.date_range is None
        assert req.trip.dates.duration_days is None
        assert req.trip.dates.flexibility == DateFlexibility.FLEXIBLE

    def test_duration_only(self) -> None:
        req = TravelRequirements(
            trip=TripDetails(
                destination="Goa",
                dates=TripDates(duration_days=4),
            ),
            travelers=TravelerDetails(count=1),
            budget=BudgetDetails(),
            hard_constraints=HardConstraints(),
            soft_preferences=SoftPreferences(),
            contextual_signals=ContextualSignals(),
        )
        assert req.trip.dates.duration_days == 4
        assert req.trip.dates.date_range is None


# ---------------------------------------------------------------------------
# 3. Approximate budget — "around ₹30,000"
# ---------------------------------------------------------------------------


class TestApproximateBudget:
    def test_approximate_budget_no_hard_constraint(self) -> None:
        req = TravelRequirements(
            trip=TripDetails(destination="Goa"),
            travelers=TravelerDetails(count=1),
            budget=BudgetDetails(
                total=Decimal("30000"),
                currency="INR",
                is_approximate=True,
            ),
            hard_constraints=HardConstraints(),
            soft_preferences=SoftPreferences(),
            contextual_signals=ContextualSignals(),
        )
        assert req.budget.is_approximate is True
        assert req.budget.total == Decimal("30000")
        assert req.hard_constraints.max_budget is None


# ---------------------------------------------------------------------------
# 4. Hard budget ceiling — "do not exceed ₹30,000"
# ---------------------------------------------------------------------------


class TestHardBudgetCeiling:
    def test_hard_max_budget(self) -> None:
        req = TravelRequirements(
            trip=TripDetails(destination="Goa"),
            travelers=TravelerDetails(count=1),
            budget=BudgetDetails(
                total=Decimal("30000"),
                currency="INR",
                is_approximate=False,
            ),
            hard_constraints=HardConstraints(max_budget=Decimal("30000")),
            soft_preferences=SoftPreferences(),
            contextual_signals=ContextualSignals(),
        )
        assert req.hard_constraints.max_budget == Decimal("30000")

    def test_hard_max_budget_without_budget_detail(self) -> None:
        """The hard constraint may be set even if budget details are sparse."""
        req = TravelRequirements(
            trip=TripDetails(destination="Goa"),
            travelers=TravelerDetails(count=1),
            budget=BudgetDetails(),
            hard_constraints=HardConstraints(max_budget=Decimal("30000")),
            soft_preferences=SoftPreferences(),
            contextual_signals=ContextualSignals(),
        )
        assert req.hard_constraints.max_budget == Decimal("30000")
        assert req.budget.total is None


# ---------------------------------------------------------------------------
# 5. Parents as contextual / companion information — no assumptions
# ---------------------------------------------------------------------------


class TestParentContext:
    def test_parents_not_seniors(self) -> None:
        """'Taking my parents to Goa' — parents are companions, not seniors."""
        req = TravelRequirements(
            trip=TripDetails(destination="Goa"),
            travelers=TravelerDetails(
                count=3,
                companions=[{"relationship": "parents"}],
            ),
            budget=BudgetDetails(),
            hard_constraints=HardConstraints(),
            soft_preferences=SoftPreferences(),
            contextual_signals=ContextualSignals(traveling_with_parents=True),
        )
        assert req.contextual_signals.traveling_with_parents is True
        assert req.travelers.companions[0].relationship == "parents"
        assert req.travelers.companions[0].categories == []
        assert TravelerCategory.SENIOR not in req.travelers.categories

    def test_no_accessibility_assumed(self) -> None:
        """Contextual parent info must not produce accessibility notes."""
        req = TravelRequirements(
            trip=TripDetails(destination="Goa"),
            travelers=TravelerDetails(
                count=3,
                companions=[{"relationship": "parents"}],
            ),
            budget=BudgetDetails(),
            hard_constraints=HardConstraints(),
            soft_preferences=SoftPreferences(),
            contextual_signals=ContextualSignals(traveling_with_parents=True),
        )
        assert req.contextual_signals.accessibility_notes is None

    def test_explicit_categories_preserved(self) -> None:
        """When the user explicitly states categories, they are stored."""
        req = TravelRequirements(
            trip=TripDetails(destination="Goa"),
            travelers=TravelerDetails(
                count=3,
                categories=[TravelerCategory.ADULT, TravelerCategory.ADULT, TravelerCategory.CHILD],
                companions=[{"relationship": "family", "categories": [TravelerCategory.ADULT, TravelerCategory.ADULT, TravelerCategory.CHILD]}],
            ),
            budget=BudgetDetails(),
            hard_constraints=HardConstraints(),
            soft_preferences=SoftPreferences(),
            contextual_signals=ContextualSignals(has_children=True),
        )
        assert len(req.travelers.categories) == 3
        assert TravelerCategory.CHILD in req.travelers.categories


# ---------------------------------------------------------------------------
# 6. ExtractionResult with clarification metadata
# ---------------------------------------------------------------------------


class TestExtractionResult:
    def test_extraction_result_with_clarifications(self) -> None:
        req = TravelRequirements(
            trip=TripDetails(destination="Goa"),
            travelers=TravelerDetails(count=2),
            budget=BudgetDetails(),
            hard_constraints=HardConstraints(),
            soft_preferences=SoftPreferences(),
            contextual_signals=ContextualSignals(),
        )
        result = ExtractionResult(
            requirements=req,
            clarification_needed=[
                ClarificationItem(
                    field="trip.origin",
                    importance=ClarificationImportance.REQUIRED,
                    question="Where will you be travelling from?",
                ),
                ClarificationItem(
                    field="budget.total",
                    importance=ClarificationImportance.RECOMMENDED,
                    question="Do you have a budget in mind?",
                    current_value=None,
                ),
            ],
            metadata=ExtractionMetadata(
                original_user_message="I want to go to Goa with my friend for 4 days",
                extraction_confidence=0.75,
                explicit_fields=["trip.destination", "trip.dates.duration_days", "travelers.count"],
                inferred_fields=[],
            ),
        )
        assert result.requirements.trip.destination == "Goa"
        assert len(result.clarification_needed) == 2
        assert result.clarification_needed[0].importance == ClarificationImportance.REQUIRED
        assert result.metadata.extraction_confidence == 0.75

    def test_extraction_result_no_clarifications(self) -> None:
        req = _complete_requirements()
        result = ExtractionResult(
            requirements=req,
            metadata=ExtractionMetadata(
                original_user_message="Book a trip to Goa from Mumbai, Dec 20-24, 3 people, budget 50k",
                extraction_confidence=0.95,
                explicit_fields=["trip.origin", "trip.destination"],
            ),
        )
        assert result.clarification_needed == []

    def test_extraction_result_round_trip(self) -> None:
        req = _complete_requirements()
        result = ExtractionResult(
            requirements=req,
            clarification_needed=[
                ClarificationItem(
                    field="trip.origin",
                    importance=ClarificationImportance.REQUIRED,
                    question="Where from?",
                ),
            ],
            metadata=ExtractionMetadata(
                original_user_message="Goa trip",
                extraction_confidence=0.6,
            ),
        )
        data = result.model_dump()
        restored = ExtractionResult.model_validate(data)
        assert restored.requirements.trip.destination == "Goa"
        assert len(restored.clarification_needed) == 1

    def test_clarification_with_interpretations(self) -> None:
        item = ClarificationItem(
            field="budget.total",
            importance=ClarificationImportance.RECOMMENDED,
            question="You mentioned a tight budget — what is the approximate amount?",
            current_value="tight",
            possible_interpretations=["under 20000 INR", "under 50000 INR"],
        )
        assert len(item.possible_interpretations) == 2


# ---------------------------------------------------------------------------
# 7. Validation — invalid numeric values
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_traveler_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TravelerDetails(count=-1)

    def test_zero_traveler_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TravelerDetails(count=0)

    def test_none_traveler_count_allowed(self) -> None:
        td = TravelerDetails(count=None)
        assert td.count is None

    def test_confidence_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExtractionMetadata(
                original_user_message="test",
                extraction_confidence=-0.1,
            )

    def test_confidence_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExtractionMetadata(
                original_user_message="test",
                extraction_confidence=1.5,
            )

    def test_valid_confidence_boundaries(self) -> None:
        low = ExtractionMetadata(original_user_message="a", extraction_confidence=0.0)
        high = ExtractionMetadata(original_user_message="b", extraction_confidence=1.0)
        assert low.extraction_confidence == 0.0
        assert high.extraction_confidence == 1.0


# ---------------------------------------------------------------------------
# 8. Enum validation
# ---------------------------------------------------------------------------


class TestEnums:
    def test_transport_modes(self) -> None:
        for mode in TransportMode:
            req = TravelRequirements(
                trip=TripDetails(destination="Goa"),
                travelers=TravelerDetails(count=1),
                budget=BudgetDetails(),
                hard_constraints=HardConstraints(required_transport=mode),
                soft_preferences=SoftPreferences(),
                contextual_signals=ContextualSignals(),
            )
            assert req.hard_constraints.required_transport == mode

    def test_comfort_levels(self) -> None:
        for level in ComfortLevel:
            req = TravelRequirements(
                trip=TripDetails(destination="Goa"),
                travelers=TravelerDetails(count=1),
                budget=BudgetDetails(),
                hard_constraints=HardConstraints(),
                soft_preferences=SoftPreferences(comfort_level=level),
                contextual_signals=ContextualSignals(),
            )
            assert req.soft_preferences.comfort_level == level

    def test_time_preferences(self) -> None:
        for pref in TimePreference:
            req = TravelRequirements(
                trip=TripDetails(destination="Goa"),
                travelers=TravelerDetails(count=1),
                budget=BudgetDetails(),
                hard_constraints=HardConstraints(departure_time=pref),
                soft_preferences=SoftPreferences(),
                contextual_signals=ContextualSignals(),
            )
            assert req.hard_constraints.departure_time == pref

    def test_date_flexibility(self) -> None:
        for flex in DateFlexibility:
            td = TripDates(flexibility=flex)
            assert td.flexibility == flex

    def test_invalid_transport_mode_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HardConstraints(required_transport="helicopter")

    def test_invalid_comfort_level_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SoftPreferences(comfort_level="ultra-premium")

    def test_traveler_categories(self) -> None:
        for cat in TravelerCategory:
            td = TravelerDetails(count=1, categories=[cat])
            assert td.categories[0] == cat
