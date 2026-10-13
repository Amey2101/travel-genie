"""TravelAgent planning integration tests (Milestone 4)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from app.agent.orchestrator import TravelAgent
from app.agent.state import AgentAction, WorkflowStatus
from app.agent.store import InMemoryWorkflowStore
from app.decision.deterministic import DeterministicDecisionEngine
from app.decision.engine import RecommendationInput, RecommendationStatus
from app.providers.demo_search import DemoSearchProvider
from app.providers.llm import (
    BudgetExtraction,
    BudgetKind,
    LLMExtractionOutput,
)
from app.providers.stub_llm import StubLLMClient
from app.services.requirement_extraction import RequirementExtractionService
from app.tools.search import (
    SearchError,
    SearchRequest,
    SearchResponse,
    SearchProvider,
)


def _complete(**kwargs) -> LLMExtractionOutput:
    base = dict(
        trip_origin="Mumbai",
        trip_destination="Goa",
        duration_days=3,
        traveler_count=2,
    )
    base.update(kwargs)
    return LLMExtractionOutput(**base)


def _goa_only() -> LLMExtractionOutput:
    return LLMExtractionOutput(trip_destination="Goa")


class _SpyProvider:
    """Records calls, delegates to the demo provider."""

    def __init__(self) -> None:
        self.calls: list[SearchRequest] = []
        self._inner = DemoSearchProvider()

    def search(self, request: SearchRequest) -> SearchResponse:
        self.calls.append(request)
        return self._inner.search(request)


class _FailingProvider:
    def __init__(self) -> None:
        self.calls: list[SearchRequest] = []

    def search(self, request: SearchRequest) -> SearchResponse:
        self.calls.append(request)
        raise SearchError("no live inventory configured")


class _EmptyProvider:
    def __init__(self) -> None:
        self.calls: list[SearchRequest] = []

    def search(self, request: SearchRequest) -> SearchResponse:
        self.calls.append(request)
        return SearchResponse(workflow_id=request.workflow_id, results=[])


class _FailingEngine:
    def recommend(self, input_: RecommendationInput):
        raise RuntimeError("boom")


def _agent(
    script: list[LLMExtractionOutput],
    *,
    provider=None,
    engine=None,
) -> TravelAgent:
    calls: list = []

    def factory(request):
        index = len(calls)
        output = script[index] if index < len(script) else LLMExtractionOutput()
        calls.append(request)
        return output

    client = StubLLMClient(factory=factory)
    service = RequirementExtractionService(
        client, reference_datetime=datetime(2026, 10, 13, tzinfo=timezone.utc)
    )
    return TravelAgent(
        extraction=service,
        store=InMemoryWorkflowStore(),
        search_provider=provider,
        decision_engine=engine,
    )


class TestPlanningTriggeredOnCompletion:
    def test_complete_requirements_run_search_and_decision(self) -> None:
        provider = _SpyProvider()
        agent = _agent(
            [_complete()], provider=provider, engine=DeterministicDecisionEngine()
        )
        workflow_id = agent.create_workflow()
        response = agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")

        assert response.status is WorkflowStatus.REQUIREMENTS_COMPLETE
        assert response.action is AgentAction.SEARCH
        assert response.search_handoff is not None
        assert len(provider.calls) == 1
        assert response.recommendation is not None
        assert response.recommendation.status is RecommendationStatus.OK
        assert response.recommendation.options

    def test_results_and_recommendation_are_persisted(self) -> None:
        provider = _SpyProvider()
        agent = _agent(
            [_complete()], provider=provider, engine=DeterministicDecisionEngine()
        )
        workflow_id = agent.create_workflow()
        agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")

        state = agent.get_state(workflow_id)
        assert state.search_results
        assert all(r.source.value == "demo" for r in state.search_results)
        assert state.recommendation is not None
        assert state.recommendation.status is RecommendationStatus.OK

    def test_planning_provider_satisfies_protocol(self) -> None:
        assert isinstance(_SpyProvider(), SearchProvider)


class TestClarificationSkipsPlanning:
    def test_missing_required_information_does_not_search(self) -> None:
        provider = _SpyProvider()
        agent = _agent(
            [_goa_only()], provider=provider, engine=DeterministicDecisionEngine()
        )
        workflow_id = agent.create_workflow()
        response = agent.handle_message(workflow_id, "Goa trip")

        assert response.status is WorkflowStatus.AWAITING_CLARIFICATION
        assert response.recommendation is None
        assert provider.calls == []
        state = agent.get_state(workflow_id)
        assert state.recommendation is None

    def test_planning_runs_only_after_clarification_resolved(self) -> None:
        provider = _SpyProvider()
        agent = _agent(
            [_goa_only(), _complete()],
            provider=provider,
            engine=DeterministicDecisionEngine(),
        )
        workflow_id = agent.create_workflow()
        agent.handle_message(workflow_id, "Goa trip")
        assert provider.calls == []

        second = agent.handle_message(workflow_id, "from Mumbai, 3 days, 2 people")
        assert second.status is WorkflowStatus.REQUIREMENTS_COMPLETE
        assert len(provider.calls) == 1
        assert second.recommendation is not None


class TestPlanningFailureHandling:
    def test_provider_unavailable_is_explicit_not_success(self) -> None:
        provider = _FailingProvider()
        agent = _agent(
            [_complete()], provider=provider, engine=DeterministicDecisionEngine()
        )
        workflow_id = agent.create_workflow()
        response = agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")

        assert response.recommendation is not None
        assert (
            response.recommendation.status
            is RecommendationStatus.PROVIDER_UNAVAILABLE
        )
        assert response.recommendation.options == []
        assert response.status is WorkflowStatus.REQUIREMENTS_COMPLETE
        assert agent.get_state(workflow_id).search_results == []

    def test_empty_results_are_explicit(self) -> None:
        provider = _EmptyProvider()
        agent = _agent(
            [_complete()], provider=provider, engine=DeterministicDecisionEngine()
        )
        workflow_id = agent.create_workflow()
        response = agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")

        assert response.recommendation is not None
        assert response.recommendation.status is RecommendationStatus.NO_RESULTS

    def test_all_candidates_rejected_is_explicit(self) -> None:
        provider = _SpyProvider()
        output = _complete(
            budget=BudgetExtraction(
                amount=Decimal("1"), currency="INR", kind=BudgetKind.HARD_MAX
            )
        )
        agent = _agent(
            [output], provider=provider, engine=DeterministicDecisionEngine()
        )
        workflow_id = agent.create_workflow()
        response = agent.handle_message(workflow_id, "Mumbai to Goa cheaply")

        assert response.recommendation is not None
        assert (
            response.recommendation.status
            is RecommendationStatus.NO_FEASIBLE_OPTIONS
        )
        assert response.recommendation.rejected

    def test_decision_engine_error_is_explicit(self) -> None:
        provider = _SpyProvider()
        agent = _agent([_complete()], provider=provider, engine=_FailingEngine())
        workflow_id = agent.create_workflow()
        response = agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")

        assert response.recommendation is not None
        assert response.recommendation.status is RecommendationStatus.DECISION_ERROR
        assert response.status is WorkflowStatus.REQUIREMENTS_COMPLETE

    def test_no_provider_configured_skips_planning(self) -> None:
        agent = _agent([_complete()])
        workflow_id = agent.create_workflow()
        response = agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")

        assert response.status is WorkflowStatus.REQUIREMENTS_COMPLETE
        assert response.search_handoff is not None
        assert response.recommendation is None

    def test_provider_failure_does_not_set_workflow_failed(self) -> None:
        agent = _agent(
            [_complete()],
            provider=_FailingProvider(),
            engine=DeterministicDecisionEngine(),
        )
        workflow_id = agent.create_workflow()
        agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")
        state = agent.get_state(workflow_id)
        assert state.status is WorkflowStatus.REQUIREMENTS_COMPLETE
        assert state.errors == []


class TestApprovalBoundaryUnchanged:
    def test_planning_creates_no_approval(self) -> None:
        provider = _SpyProvider()
        agent = _agent(
            [_complete()], provider=provider, engine=DeterministicDecisionEngine()
        )
        workflow_id = agent.create_workflow()
        agent.handle_message(workflow_id, "Mumbai to Goa 3 days 2 people")
        state = agent.get_state(workflow_id)
        assert not hasattr(state, "approval")

    def test_execution_still_requires_explicit_approval(self) -> None:
        from pydantic import ValidationError

        from app.agent.security import ConsequentialAction
        from app.tools.execution import ExecutionRequest

        with pytest.raises(ValidationError):
            ExecutionRequest(
                workflow_id="wf-1",
                action=ConsequentialAction.BOOK,
                approval=None,
            )
