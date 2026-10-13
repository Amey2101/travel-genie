"""Unit tests for the deterministic structured-input adapter."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.models.travel_requirements import ClarificationImportance
from app.services.structured_requirements import (
    ClarificationAnswer,
    ClarificationError,
    StructuredRequirementsRequest,
    build_extraction_result,
    merge_clarification,
)


def _base(**kwargs) -> StructuredRequirementsRequest:
    data = dict(
        origin="Pune",
        destination="Goa",
        departure_date=date(2026, 11, 10),
        return_date=date(2026, 11, 12),
        travelers=2,
        budget=Decimal("25000"),
        currency="INR",
    )
    data.update(kwargs)
    return StructuredRequirementsRequest(**data)


class TestMapping:
    def test_return_date_maps_to_date_range(self) -> None:
        result = build_extraction_result(_base())
        trip = result.requirements.trip
        assert trip.origin == "Pune" and trip.destination == "Goa"
        assert trip.dates is not None
        assert trip.dates.date_range.start == date(2026, 11, 10)
        assert trip.dates.date_range.end == date(2026, 11, 12)
        assert trip.dates.duration_days is None

    def test_duration_only_maps_to_duration(self) -> None:
        req = _base(return_date=None, duration_days=4)
        result = build_extraction_result(req)
        dates = result.requirements.trip.dates
        assert dates is not None
        assert dates.date_range is None
        assert dates.duration_days == 4

    def test_travelers_and_interests_and_notes_preserved(self) -> None:
        req = _base(
            interests=["food", "beaches", "food"],
            accommodation="beach guesthouse",
            notes="vegetarian",
        )
        result = build_extraction_result(req)
        assert result.requirements.travelers.count == 2
        assert result.requirements.soft_preferences.priorities == ["food", "beaches"]
        notes = result.requirements.contextual_signals.other_notes
        assert any("beach guesthouse" in n for n in notes)
        assert "vegetarian" in notes

    def test_travel_style_maps_to_comfort_level(self) -> None:
        assert (
            build_extraction_result(_base(travel_style="budget"))
            .requirements.soft_preferences.comfort_level.value
            == "budget"
        )
        assert (
            build_extraction_result(_base(travel_style="luxury"))
            .requirements.soft_preferences.comfort_level.value
            == "luxury"
        )


class TestBudgetSemantics:
    def test_stated_budget_is_approximate_not_hard_max(self) -> None:
        result = build_extraction_result(_base(budget=Decimal("25000")))
        budget = result.requirements.budget
        assert budget.total == Decimal("25000")
        assert budget.is_approximate is True
        assert result.requirements.hard_constraints.max_budget is None


class TestTemporalClarification:
    def test_departure_without_duration_or_return_needs_timing(self) -> None:
        req = _base(return_date=None, duration_days=None)
        result = build_extraction_result(req)
        required = [
            item
            for item in result.clarification_needed
            if item.importance is ClarificationImportance.REQUIRED
        ]
        assert any(item.field == "travel.timing" for item in required)


class TestValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"origin": ""},
            {"destination": ""},
            {"return_date": date(2026, 11, 1)},
            {"return_date": date(2026, 11, 12), "duration_days": 5},
            {"currency": "XYZ"},
            {"interests": ["skiing"]},
            {"travelers": 0},
            {"duration_days": 0},
        ],
    )
    def test_invalid_inputs_rejected(self, overrides: dict) -> None:
        with pytest.raises(ValidationError):
            _base(**overrides)


class TestSameDayTrip:
    def test_same_day_return_is_allowed(self) -> None:
        result = build_extraction_result(_base(return_date=date(2026, 11, 10)))
        dates = result.requirements.trip.dates
        assert dates is not None
        assert dates.date_range.start == dates.date_range.end == date(2026, 11, 10)


class TestClarificationAnswer:
    def test_return_and_duration_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValidationError):
            ClarificationAnswer(
                return_date=date(2026, 11, 12), duration_days=3
            )

    def test_return_before_departure_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ClarificationAnswer(
                departure_date=date(2026, 11, 10),
                return_date=date(2026, 11, 1),
            )

    def test_bad_currency_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ClarificationAnswer(currency="XYZ")

    def test_text_is_stripped(self) -> None:
        answer = ClarificationAnswer(origin="  Pune  ", destination=" Goa ")
        assert answer.origin == "Pune"
        assert answer.destination == "Goa"


class TestMergeClarification:
    def _timing_incomplete(self):
        return build_extraction_result(
            _base(return_date=None, duration_days=None, interests=["food"])
        )

    def test_duration_answer_completes_timing_and_preserves_context(self) -> None:
        base = self._timing_incomplete()
        merged = merge_clarification(base, ClarificationAnswer(duration_days=3))
        trip = merged.requirements.trip
        assert trip.origin == "Pune" and trip.destination == "Goa"
        assert trip.dates is not None
        assert trip.dates.duration_days == 3
        assert trip.dates.date_range is None
        assert merged.requirements.soft_preferences.priorities == ["food"]
        assert merged.requirements.budget.total == Decimal("25000")
        assert merged.requirements.budget.is_approximate is True

    def test_return_date_answer_creates_date_range(self) -> None:
        base = self._timing_incomplete()
        merged = merge_clarification(
            base,
            ClarificationAnswer(
                departure_date=date(2026, 11, 10),
                return_date=date(2026, 11, 12),
            ),
        )
        dates = merged.requirements.trip.dates
        assert dates.date_range.start == date(2026, 11, 10)
        assert dates.date_range.end == date(2026, 11, 12)
        assert dates.duration_days is None

    def test_explicit_budget_overrides_but_stays_approximate(self) -> None:
        base = self._timing_incomplete()
        merged = merge_clarification(
            base,
            ClarificationAnswer(duration_days=3, budget=Decimal("50000")),
        )
        assert merged.requirements.budget.total == Decimal("50000")
        assert merged.requirements.budget.is_approximate is True
        assert merged.requirements.hard_constraints.max_budget is None

    def test_return_without_departure_is_a_controlled_error(self) -> None:
        base = self._timing_incomplete()
        with pytest.raises(ClarificationError):
            merge_clarification(
                base, ClarificationAnswer(return_date=date(2026, 11, 12))
            )
