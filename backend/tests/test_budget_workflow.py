"""TravelAgent budget-estimation integration tests (Milestone 6A)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.agent.orchestrator import TravelAgent
from app.agent.state import WorkflowStatus
from app.agent.store import InMemoryWorkflowStore
from app.decision.budget import StyleBasedBudgetEstimator
from app.decision.deterministic import DeterministicDecisionEngine
from app.decision.engine import (
    Recommendation,
    RecommendationInput,
    RecommendationStatus,
)
from app.main import app
from app.providers.demo_search import DemoSearchProvider
from app.providers.llm import BudgetExtraction, BudgetKind, LLMExtractionOutput
from app.providers.stub_llm import StubLLMClient
from app.services.requirement_extraction import RequirementExtractionService
from app.tools.search import SearchError, SearchRequest, SearchResponse


def _complete(**kwargs) -> LLMExtractionOutput:
    base = dict(
        trip_origin="Mumbai",
        trip_destination="Goa",
        duration_days=3,
        traveler_count=2,
    )
    base.update(kwargs)
    return LLMExtractionOutput(**base)


class _SpyProvider:
    def __init__(self) -> None:
        self._inner = DemoSearchProvider()

    def search(self, request: SearchRequest) -> SearchResponse:
        return self._inner.search(request)


def _agent(
    output: LLMExtractionOutput,
    *,
    budget_estimator=None,
    search_provider=None,
    decision_engine=None,
) -> TravelAgent:
    calls: list = []

    def factory(request):
        # Every turn re-extracts the same complete requirements so multi-turn
        # planning can be exercised deterministically.
        calls.append(request)
        return output

    client = StubLLMClient(factory=factory)
    service = RequirementExtractionService(
        client, reference_datetime=datetime(2026, 10, 13, tzinfo=timezone.utc)
    )
    return TravelAgent(
        extraction=service,
        store=InMemoryWorkflowStore(),
        search_provider=search_provider or _SpyProvider(),
        decision_engine=decision_engine or DeterministicDecisionEngine(),
        budget_estimator=budget_estimator,
    )


class _FlakyProvider:
    """Search succeeds on the first call, then fails like a down provider."""

    def __init__(self, failures: int = 1) -> None:
        self._inner = DemoSearchProvider()
        self._calls = 0
        self._failures = failures

    def search(self, request: SearchRequest) -> SearchResponse:
        self._calls += 1
        if self._calls > 1 and self._calls - 1 <= self._failures:
            raise SearchError("provider down")
        return self._inner.search(request)


class _FlakyEngine:
    """Decision engine succeeds on the first call, then raises."""

    def __init__(self) -> None:
        self._inner = DeterministicDecisionEngine()
        self._calls = 0

    def recommend(self, input_: RecommendationInput) -> Recommendation:
        self._calls += 1
        if self._calls > 1:
            raise RuntimeError("decision engine bug")
        return self._inner.recommend(input_)


class TestWorkflowBudgetIntegration:
    def test_estimate_attached_and_persisted(self) -> None:
        agent = _agent(_complete(), budget_estimator=StyleBasedBudgetEstimator())
        workflow_id = agent.create_workflow()
        response = agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")

        assert response.status is WorkflowStatus.REQUIREMENTS_COMPLETE
        assert response.recommendation is not None
        assert response.recommendation.status is RecommendationStatus.OK
        assert response.budget_estimate is not None
        assert response.budget_estimate.estimated_total > 0

        state = agent.get_state(workflow_id)
        assert state.budget_estimate is not None
        assert state.budget_estimate == response.budget_estimate

    def test_demo_prices_are_not_used_as_fares(self) -> None:
        agent = _agent(_complete(), budget_estimator=StyleBasedBudgetEstimator())
        workflow_id = agent.create_workflow()
        response = agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")

        assert response.budget_estimate is not None
        assert response.budget_estimate.transportation_source.value == "heuristic"
        assert state_demo_sources(agent, workflow_id)

    def test_recommendation_behaviour_unchanged(self) -> None:
        agent = _agent(_complete(), budget_estimator=StyleBasedBudgetEstimator())
        workflow_id = agent.create_workflow()
        response = agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")

        assert response.recommendation is not None
        assert response.recommendation.status is RecommendationStatus.OK
        assert response.recommendation.options
        assert all(
            r.source.value == "demo" and r.bookable is False
            for r in agent.get_state(workflow_id).search_results
        )

    def test_no_estimator_wired_leaves_estimate_absent(self) -> None:
        agent = _agent(_complete())
        workflow_id = agent.create_workflow()
        response = agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")

        assert response.recommendation is not None
        assert response.budget_estimate is None
        assert agent.get_state(workflow_id).budget_estimate is None

    def test_missing_travelers_leaves_estimate_absent(self) -> None:
        output = _complete(traveler_count=None)
        agent = _agent(output, budget_estimator=StyleBasedBudgetEstimator())
        workflow_id = agent.create_workflow()
        response = agent.handle_message(workflow_id, "Mumbai to Goa 3 days")

        # Requirements still complete (travelers is only recommended), but no
        # estimate is fabricated without a traveller count.
        assert response.status is WorkflowStatus.REQUIREMENTS_COMPLETE
        assert response.budget_estimate is None

    def test_approximate_budget_not_treated_as_hard_constraint(self) -> None:
        output = _complete(
            budget=BudgetExtraction(
                amount=Decimal("25000"), currency="INR", kind=BudgetKind.APPROXIMATE
            )
        )
        agent = _agent(output, budget_estimator=StyleBasedBudgetEstimator())
        workflow_id = agent.create_workflow()
        response = agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")

        assert response.budget_estimate is not None
        assert response.budget_estimate.target_budget == Decimal("25000")
        assert response.budget_estimate.target_is_approximate is True
        assert response.budget_estimate.hard_max_budget is None


class TestStaleEstimateIsCleared:
    """A prior turn's estimate must never survive a failed planning attempt."""

    def test_provider_failure_after_success_leaves_no_stale_estimate(self) -> None:
        agent = _agent(
            _complete(),
            budget_estimator=StyleBasedBudgetEstimator(),
            search_provider=_FlakyProvider(failures=1),
        )
        workflow_id = agent.create_workflow()

        first = agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")
        assert first.recommendation is not None
        assert first.recommendation.status is RecommendationStatus.OK
        assert first.budget_estimate is not None

        second = agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")
        assert second.recommendation is not None
        assert second.recommendation.status is RecommendationStatus.PROVIDER_UNAVAILABLE
        assert second.budget_estimate is None

        state = agent.get_state(workflow_id)
        assert state.recommendation is not None
        assert state.recommendation.status is RecommendationStatus.PROVIDER_UNAVAILABLE
        assert state.budget_estimate is None

    def test_decision_error_after_success_leaves_no_stale_estimate(self) -> None:
        agent = _agent(
            _complete(),
            budget_estimator=StyleBasedBudgetEstimator(),
            decision_engine=_FlakyEngine(),
        )
        workflow_id = agent.create_workflow()

        first = agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")
        assert first.budget_estimate is not None

        second = agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")
        assert second.recommendation is not None
        assert second.recommendation.status is RecommendationStatus.DECISION_ERROR
        assert second.budget_estimate is None

        state = agent.get_state(workflow_id)
        assert state.recommendation.status is RecommendationStatus.DECISION_ERROR
        assert state.budget_estimate is None

    def test_subsequent_successful_planning_yields_a_fresh_estimate(self) -> None:
        agent = _agent(
            _complete(),
            budget_estimator=StyleBasedBudgetEstimator(),
            search_provider=_FlakyProvider(failures=1),
        )
        workflow_id = agent.create_workflow()

        first = agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")
        assert first.budget_estimate is not None

        second = agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")
        assert second.budget_estimate is None

        third = agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")
        assert third.recommendation is not None
        assert third.recommendation.status is RecommendationStatus.OK
        assert third.budget_estimate is not None

        state = agent.get_state(workflow_id)
        assert state.budget_estimate is not None
        assert state.budget_estimate == third.budget_estimate

    def test_no_estimator_wired_still_clears_without_error(self) -> None:
        agent = _agent(_complete(), search_provider=_FlakyProvider(failures=1))
        workflow_id = agent.create_workflow()
        agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")
        second = agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")
        assert second.recommendation is not None
        assert second.recommendation.status is RecommendationStatus.PROVIDER_UNAVAILABLE
        assert agent.get_state(workflow_id).budget_estimate is None


class TestUnsupportedCurrencyEndToEnd:
    def test_free_text_path_yields_no_estimate_for_unsupported_currency(self) -> None:
        output = _complete(
            budget=BudgetExtraction(
                amount=Decimal("50000"), currency="JPY", kind=BudgetKind.APPROXIMATE
            )
        )
        agent = _agent(output, budget_estimator=StyleBasedBudgetEstimator())
        workflow_id = agent.create_workflow()
        response = agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")

        assert response.recommendation is not None
        assert response.budget_estimate is None
        assert agent.get_state(workflow_id).budget_estimate is None

    def test_lowercase_supported_currency_produces_an_estimate(self) -> None:
        output = _complete(
            budget=BudgetExtraction(
                amount=Decimal("40000"), currency="usd", kind=BudgetKind.APPROXIMATE
            )
        )
        agent = _agent(output, budget_estimator=StyleBasedBudgetEstimator())
        workflow_id = agent.create_workflow()
        response = agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")

        assert response.budget_estimate is not None
        assert response.budget_estimate.currency == "USD"


def state_demo_sources(agent: TravelAgent, workflow_id: str) -> bool:
    results = agent.get_state(workflow_id).search_results
    return all(r.source.value == "demo" and r.bookable is False for r in results)


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    for key in ("GEMINI_API_KEY", "GROQ_API_KEY", "SERPAPI_API_KEY"):
        monkeypatch.delenv(key, raising=False)


class TestWorkflowApiBudgetSerialization:
    def test_get_workflow_exposes_budget_estimate(self) -> None:
        client = TestClient(app)
        workflow_id = client.post("/workflows").json()["workflow_id"]
        response = client.post(
            f"/workflows/{workflow_id}/requirements",
            json={
                "origin": "Pune",
                "destination": "Goa",
                "departure_date": "2026-12-20",
                "return_date": "2026-12-24",
                "travelers": 2,
                "budget": 25000,
                "currency": "INR",
                "travel_style": "balanced",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["budget_estimate"] is not None
        assert body["budget_estimate"]["currency"] == "INR"
        assert body["budget_estimate"]["target_budget"] == "25000"

        persisted = client.get(f"/workflows/{workflow_id}")
        assert persisted.status_code == 200
        assert persisted.json()["budget_estimate"] == body["budget_estimate"]

    def test_budget_estimate_has_category_lines(self) -> None:
        client = TestClient(app)
        workflow_id = client.post("/workflows").json()["workflow_id"]
        response = client.post(
            f"/workflows/{workflow_id}/requirements",
            json={
                "origin": "Pune",
                "destination": "Goa",
                "departure_date": "2026-12-20",
                "duration_days": 4,
                "travelers": 2,
                "budget": 25000,
                "currency": "INR",
                "travel_style": "budget",
            },
        )
        estimate = response.json()["budget_estimate"]
        assert estimate is not None
        keys = [line["key"] for line in estimate["lines"]]
        assert keys == [
            "transportation",
            "accommodation",
            "food",
            "activities",
            "miscellaneous",
        ]
        assert estimate["disclaimer"]
