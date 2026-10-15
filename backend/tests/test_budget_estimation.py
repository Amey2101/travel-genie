"""Deterministic budget-estimation tests (Milestone 6A).

These exercise the estimator contract and the explicit mapping from our domain
model; they do not touch the network or an LLM.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.decision.budget import (
    BudgetEstimateInput,
    BudgetEstimator,
    StyleBasedBudgetEstimator,
    StyleTier,
    TransportationSource,
    budget_input_from_requirements,
    trip_days_and_nights,
)
from app.decision.engine import (
    Candidate,
    Recommendation,
    RecommendationStatus,
    RecommendedOption,
)
from app.models.travel_requirements import (
    BudgetDetails,
    ComfortLevel,
    ContextualSignals,
    DateRange,
    HardConstraints,
    SoftPreferences,
    TravelerDetails,
    TravelRequirements,
    TripDates,
    TripDetails,
)
from app.tools.search import ResultSource


def _input(**kwargs) -> BudgetEstimateInput:
    base = dict(
        origin="Pune",
        destination="Goa",
        currency="INR",
        travelers=2,
        days=4,
        nights=3,
        style=StyleTier.BALANCED,
    )
    base.update(kwargs)
    return BudgetEstimateInput(**base)


_DEFAULT = object()


def _requirements(
    *,
    origin="Pune",
    destination="Goa",
    dates=_DEFAULT,
    travelers=2,
    total=None,
    is_approximate=False,
    hard_max=None,
    comfort=ComfortLevel.STANDARD,
    currency="INR",
) -> TravelRequirements:
    if dates is _DEFAULT:
        dates = TripDates(duration_days=4)
    return TravelRequirements(
        trip=TripDetails(origin=origin, destination=destination, dates=dates),
        travelers=TravelerDetails(count=travelers),
        budget=BudgetDetails(total=total, currency=currency, is_approximate=is_approximate),
        hard_constraints=HardConstraints(max_budget=hard_max),
        soft_preferences=SoftPreferences(comfort_level=comfort),
        contextual_signals=ContextualSignals(),
    )


def _option(
    *,
    price="12000",
    currency="INR",
    source=ResultSource.VERIFIED,
    transport_mode=None,
) -> Recommendation:
    from app.models.travel_requirements import TransportMode

    candidate = Candidate(
        provider_ref="cand-1",
        label="Test option",
        price=Decimal(price) if price is not None else None,
        currency=currency,
        source=source,
        transport_mode=transport_mode or TransportMode.FLIGHT,
    )
    return Recommendation(
        workflow_id="wf-1",
        status=RecommendationStatus.OK,
        options=[RecommendedOption(rank=1, candidate=candidate, score=1.0)],
    )


class TestEstimatorContract:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(StyleBasedBudgetEstimator(), BudgetEstimator)

    def test_deterministic(self) -> None:
        estimator = StyleBasedBudgetEstimator()
        first = estimator.estimate(_input())
        second = estimator.estimate(_input())
        assert first == second

    def test_total_is_exact_sum_of_lines(self) -> None:
        estimate = StyleBasedBudgetEstimator().estimate(_input())
        assert estimate.estimated_total == sum(l.amount for l in estimate.lines)

    def test_category_calculation(self) -> None:
        estimate = StyleBasedBudgetEstimator().estimate(_input())
        amounts = {l.key: l.amount for l in estimate.lines}
        assert amounts == {
            "transportation": Decimal("9800"),
            "accommodation": Decimal("7500"),
            "food": Decimal("6400"),
            "activities": Decimal("4800"),
            "miscellaneous": Decimal("2280"),
        }
        assert estimate.estimated_total == Decimal("30780")

    def test_same_city_drops_intercity_cost(self) -> None:
        same = StyleBasedBudgetEstimator().estimate(
            _input(origin="Goa", destination="goa")
        )
        # balanced local_day 700 * 4 days, no intercity leg
        transport = next(l for l in same.lines if l.key == "transportation")
        assert transport.amount == Decimal("2800")


class TestTravelStyles:
    def test_tiers_are_ordered(self) -> None:
        estimator = StyleBasedBudgetEstimator()
        budget = estimator.estimate(_input(style=StyleTier.BUDGET)).estimated_total
        balanced = estimator.estimate(_input(style=StyleTier.BALANCED)).estimated_total
        luxury = estimator.estimate(_input(style=StyleTier.LUXURY)).estimated_total
        assert budget < balanced < luxury
        assert (budget, balanced, luxury) == (
            Decimal("16850"),
            Decimal("30780"),
            Decimal("87480"),
        )


class TestBudgetSemantics:
    def test_missing_target_is_explicit(self) -> None:
        estimate = StyleBasedBudgetEstimator().estimate(_input(target_budget=None))
        assert estimate.target_budget is None
        assert estimate.difference is None
        assert estimate.within_target is None
        assert any("No target budget" in s for s in estimate.suggestions)

    def test_below_target(self) -> None:
        estimate = StyleBasedBudgetEstimator().estimate(
            _input(target_budget=Decimal("40000"))
        )
        assert estimate.within_target is True
        assert estimate.difference == Decimal("9220")
        assert any("buffer" in s for s in estimate.suggestions)

    def test_above_target(self) -> None:
        estimate = StyleBasedBudgetEstimator().estimate(
            _input(target_budget=Decimal("20000"))
        )
        assert estimate.within_target is False
        assert estimate.difference == Decimal("-10780")
        assert any("over your target" in s for s in estimate.suggestions)

    def test_approximate_target_is_not_a_hard_constraint(self) -> None:
        estimate = StyleBasedBudgetEstimator().estimate(
            _input(
                target_budget=Decimal("20000"),
                target_is_approximate=True,
                hard_max_budget=None,
            )
        )
        assert estimate.target_is_approximate is True
        assert estimate.hard_max_budget is None
        assert estimate.hard_max_exceeded is None

    def test_hard_max_is_distinct_from_target(self) -> None:
        estimate = StyleBasedBudgetEstimator().estimate(
            _input(
                target_budget=Decimal("40000"),
                target_is_approximate=False,
                hard_max_budget=Decimal("25000"),
            )
        )
        assert estimate.target_budget == Decimal("40000")
        assert estimate.hard_max_budget == Decimal("25000")
        assert estimate.within_target is True
        assert estimate.hard_max_exceeded is True
        assert any("maximum budget" in s for s in estimate.suggestions)

    def test_estimate_is_never_a_verified_price(self) -> None:
        estimate = StyleBasedBudgetEstimator().estimate(_input())
        assert estimate.transportation_source is TransportationSource.HEURISTIC
        assert "not verified" in estimate.disclaimer


class TestDateAndDurationSemantics:
    def test_date_range_counts_return_day_inclusively(self) -> None:
        dates = TripDates(
            date_range=DateRange(start=date(2026, 12, 20), end=date(2026, 12, 24))
        )
        assert trip_days_and_nights(dates) == (5, 4)

    def test_single_day_range(self) -> None:
        day = date(2026, 12, 20)
        dates = TripDates(date_range=DateRange(start=day, end=day))
        assert trip_days_and_nights(dates) == (1, 0)

    def test_duration_only(self) -> None:
        assert trip_days_and_nights(TripDates(duration_days=3)) == (3, 2)

    def test_no_timing_returns_none(self) -> None:
        assert trip_days_and_nights(TripDates()) is None
        assert trip_days_and_nights(None) is None


class TestCandidateProvenance:
    def test_verified_candidate_price_used_as_transportation(self) -> None:
        estimator = StyleBasedBudgetEstimator()
        plan_input = budget_input_from_requirements(
            _requirements(), _option(price="12000", source=ResultSource.VERIFIED)
        )
        assert plan_input is not None
        estimate = estimator.estimate(plan_input)
        transport = next(l for l in estimate.lines if l.key == "transportation")
        assert transport.amount == Decimal("12000")
        assert estimate.transportation_source is TransportationSource.PROVIDER_VERIFIED

    def test_demo_candidate_price_is_never_used_as_a_fare(self) -> None:
        estimator = StyleBasedBudgetEstimator()
        plan_input = budget_input_from_requirements(
            _requirements(), _option(price="12000", source=ResultSource.DEMO)
        )
        assert plan_input is not None
        assert plan_input.transportation_price is None
        estimate = estimator.estimate(plan_input)
        assert estimate.transportation_source is TransportationSource.HEURISTIC

    def test_estimate_candidate_price_is_never_used_as_a_fare(self) -> None:
        plan_input = budget_input_from_requirements(
            _requirements(), _option(price="12000", source=ResultSource.ESTIMATE)
        )
        assert plan_input is not None
        assert plan_input.transportation_price is None

    def test_candidate_without_transport_mode_is_ignored(self) -> None:
        recommendation = _option(price="12000", source=ResultSource.VERIFIED)
        recommendation.options[0].candidate.transport_mode = None
        plan_input = budget_input_from_requirements(_requirements(), recommendation)
        assert plan_input is not None
        assert plan_input.transportation_price is None


class TestDomainMapping:
    def test_maps_requirements_with_range(self) -> None:
        req = _requirements(
            dates=TripDates(
                date_range=DateRange(start=date(2026, 12, 20), end=date(2026, 12, 24))
            ),
            total=Decimal("40000"),
            is_approximate=True,
            comfort=ComfortLevel.LUXURY,
        )
        plan_input = budget_input_from_requirements(req)
        assert plan_input is not None
        assert plan_input.days == 5
        assert plan_input.nights == 4
        assert plan_input.style is StyleTier.LUXURY
        assert plan_input.target_budget == Decimal("40000")
        assert plan_input.target_is_approximate is True
        assert plan_input.transportation_price is None

    def test_maps_duration_only(self) -> None:
        req = _requirements(dates=TripDates(duration_days=3))
        plan_input = budget_input_from_requirements(req)
        assert plan_input is not None
        assert (plan_input.days, plan_input.nights) == (3, 2)

    def test_comfort_level_mapping(self) -> None:
        expected = {
            ComfortLevel.BUDGET: StyleTier.BUDGET,
            ComfortLevel.ECONOMY: StyleTier.BUDGET,
            ComfortLevel.STANDARD: StyleTier.BALANCED,
            ComfortLevel.PREMIUM: StyleTier.LUXURY,
            ComfortLevel.LUXURY: StyleTier.LUXURY,
            None: StyleTier.BALANCED,
        }
        for comfort, tier in expected.items():
            req = _requirements(dates=TripDates(duration_days=3), comfort=comfort)
            plan_input = budget_input_from_requirements(req)
            assert plan_input is not None
            assert plan_input.style is tier

    def test_missing_travelers_yields_no_estimate(self) -> None:
        req = _requirements(dates=TripDates(duration_days=3), travelers=None)
        assert budget_input_from_requirements(req) is None

    def test_missing_timing_yields_no_estimate(self) -> None:
        req = _requirements(dates=None)
        assert budget_input_from_requirements(req) is None


class TestCurrencyHandling:
    def test_unsupported_currency_yields_no_estimate(self) -> None:
        req = _requirements(currency="JPY")
        assert budget_input_from_requirements(req) is None

    def test_lowercase_and_padded_currency_is_normalised(self) -> None:
        req = _requirements(currency=" usd ", total=Decimal("40000"))
        plan_input = budget_input_from_requirements(req)
        assert plan_input is not None
        assert plan_input.currency == "USD"
        # Normalisation must not disturb any other field.
        assert plan_input.target_budget == Decimal("40000")
        assert plan_input.travelers == 2
        assert (plan_input.days, plan_input.nights) == (4, 3)

    def test_lower_case_supported_currency_matches_upper_case(self) -> None:
        upper = budget_input_from_requirements(_requirements(currency="USD"))
        lower = budget_input_from_requirements(_requirements(currency="usd"))
        assert upper is not None and lower is not None
        assert (
            StyleBasedBudgetEstimator().estimate(upper)
            == StyleBasedBudgetEstimator().estimate(lower)
        )

    def test_supported_currency_calculation_is_unchanged(self) -> None:
        estimate = StyleBasedBudgetEstimator().estimate(_input(currency="USD"))
        assert {l.key: l.amount for l in estimate.lines} == {
            "transportation": Decimal("118"),
            "accommodation": Decimal("90"),
            "food": Decimal("77"),
            "activities": Decimal("58"),
            "miscellaneous": Decimal("27"),
        }
        assert estimate.estimated_total == Decimal("370")
        assert estimate.currency == "USD"

    def test_inr_calculation_is_unchanged(self) -> None:
        estimate = StyleBasedBudgetEstimator().estimate(_input(currency="INR"))
        assert estimate.estimated_total == Decimal("30780")

    def test_estimator_rejects_unsupported_currency(self) -> None:
        # No silent 1:1 INR fallback and no INR rounding rules applied.
        with pytest.raises(ValueError, match="Unsupported currency"):
            StyleBasedBudgetEstimator().estimate(_input(currency="JPY"))

    def test_supported_currency_codes_are_accepted(self) -> None:
        supported = ["INR", "USD", "EUR", "GBP", "AED", "SGD", "THB"]
        for code in supported:
            plan_input = budget_input_from_requirements(_requirements(currency=code))
            assert plan_input is not None, code
            assert StyleBasedBudgetEstimator().estimate(plan_input).currency == code
