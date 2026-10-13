"""Tests for the deterministic demo ``SearchProvider`` (Milestone 4)."""

from __future__ import annotations

from decimal import Decimal

from app.models.travel_requirements import (
    BudgetDetails,
    ComfortLevel,
    ContextualSignals,
    HardConstraints,
    SoftPreferences,
    TransportMode,
    TravelerDetails,
    TravelRequirements,
    TripDetails,
)
from app.providers.demo_search import DemoSearchProvider
from app.tools.discovery import DiscoveryProvider
from app.tools.search import ResultSource, SearchProvider, SearchRequest


def _requirements(
    *,
    origin: str = "Mumbai",
    destination: str = "Goa",
    budget_total: Decimal | None = None,
    currency: str = "INR",
) -> TravelRequirements:
    return TravelRequirements(
        trip=TripDetails(origin=origin, destination=destination),
        travelers=TravelerDetails(count=2),
        budget=BudgetDetails(total=budget_total, currency=currency),
        hard_constraints=HardConstraints(),
        soft_preferences=SoftPreferences(),
        contextual_signals=ContextualSignals(),
    )


def _request(requirements: TravelRequirements, workflow_id: str = "wf-1") -> SearchRequest:
    return SearchRequest(workflow_id=workflow_id, requirements=requirements)


class TestDemoProviderContract:
    def test_satisfies_search_provider_protocol(self) -> None:
        assert isinstance(DemoSearchProvider(), SearchProvider)

    def test_is_not_a_discovery_provider(self) -> None:
        assert not isinstance(DemoSearchProvider(), DiscoveryProvider)

    def test_search_returns_workflow_id(self) -> None:
        response = DemoSearchProvider().search(_request(_requirements(), "wf-42"))
        assert response.workflow_id == "wf-42"
        assert response.results


class TestDemoLabelling:
    def test_every_result_is_demo_and_not_bookable(self) -> None:
        response = DemoSearchProvider().search(_request(_requirements()))
        assert response.results
        for result in response.results:
            assert result.source is ResultSource.DEMO
            assert result.bookable is False

    def test_no_result_claims_verified_status(self) -> None:
        response = DemoSearchProvider().search(_request(_requirements()))
        assert all(r.source is not ResultSource.VERIFIED for r in response.results)

    def test_results_are_categorised_as_transport_not_attractions(self) -> None:
        response = DemoSearchProvider().search(_request(_requirements()))
        assert all(r.category == "transport" for r in response.results)
        assert all(r.transport_mode is not None for r in response.results)


class TestDemoDeterminism:
    def test_repeated_equivalent_requests_are_identical(self) -> None:
        provider = DemoSearchProvider()
        first = provider.search(_request(_requirements()))
        second = provider.search(_request(_requirements()))
        assert first.model_dump() == second.model_dump()

    def test_equivalent_case_and_whitespace_are_stable(self) -> None:
        provider = DemoSearchProvider()
        first = provider.search(
            _request(_requirements(origin="Mumbai", destination="Goa"))
        )
        second = provider.search(
            _request(_requirements(origin="  mumbai ", destination="goa  "))
        )
        assert [r.provider_ref for r in first.results] == [
            r.provider_ref for r in second.results
        ]

    def test_different_routes_produce_different_references(self) -> None:
        provider = DemoSearchProvider()
        goa = provider.search(_request(_requirements(destination="Goa")))
        delhi = provider.search(_request(_requirements(destination="Delhi")))
        assert {r.provider_ref for r in goa.results}.isdisjoint(
            {r.provider_ref for r in delhi.results}
        )


class TestDemoPricesAreIndependentOfBudget:
    def test_prices_ignore_the_user_budget(self) -> None:
        provider = DemoSearchProvider()
        no_budget = provider.search(_request(_requirements()))
        huge_budget = provider.search(
            _request(_requirements(budget_total=Decimal("1000000")))
        )
        assert [r.price for r in no_budget.results] == [
            r.price for r in huge_budget.results
        ]

    def test_small_budget_does_not_shrink_prices(self) -> None:
        provider = DemoSearchProvider()
        plain = provider.search(_request(_requirements()))
        tiny_budget = provider.search(
            _request(_requirements(budget_total=Decimal("1")))
        )
        assert [r.price for r in plain.results] == [
            r.price for r in tiny_budget.results
        ]
        # And the cheapest option is clearly above the tiny budget.
        assert min(r.price for r in tiny_budget.results) > Decimal("1")


class TestDemoFilterableAttributes:
    def test_results_carry_constraint_attributes(self) -> None:
        response = DemoSearchProvider().search(_request(_requirements()))
        assert any(r.transport_mode is TransportMode.FLIGHT for r in response.results)
        for result in response.results:
            assert result.duration_days is not None
            assert result.departure_time is not None
            assert result.arrival_time is not None
            assert result.start_date is not None
            assert result.end_date is not None

    def test_currency_defaults_to_inr_and_honours_request(self) -> None:
        provider = DemoSearchProvider()
        default = provider.search(_request(_requirements()))
        assert all(r.currency == "INR" for r in default.results)
        usd = provider.search(_request(_requirements(currency="USD")))
        assert all(r.currency == "USD" for r in usd.results)


class TestDemoReplaceable:
    def test_soft_preferences_do_not_change_fixtures(self) -> None:
        base = _requirements()
        preferred = base.model_copy(
            update={
                "soft_preferences": SoftPreferences(
                    comfort_level=ComfortLevel.LUXURY,
                    preferred_transport=TransportMode.TRAIN,
                )
            }
        )
        provider = DemoSearchProvider()
        assert provider.search(_request(base)).model_dump() == provider.search(
            _request(preferred)
        ).model_dump()
