from __future__ import annotations

from decimal import Decimal

import pytest

from app.decision.engine import (
    Candidate,
    DecisionEngine,
    Recommendation,
    RecommendationInput,
    RecommendedOption,
)
from app.tools.search import SearchProvider, SearchRequest, SearchResponse, SearchResult
from app.tools.execution import ExecutionProvider, ExecutionRequest, ExecutionResult
from app.agent.security import Approval, ApprovalDecision, ConsequentialAction


# ---------------------------------------------------------------------------
# Search contracts
# ---------------------------------------------------------------------------


class TestSearchContracts:
    def test_search_request_round_trip(self) -> None:
        from app.models.travel_requirements import (
            BudgetDetails,
            ContextualSignals,
            HardConstraints,
            SoftPreferences,
            TravelerDetails,
            TravelRequirements,
            TripDetails,
        )

        req = SearchRequest(
            workflow_id="wf-1",
            requirements=TravelRequirements(
                trip=TripDetails(destination="Goa"),
                travelers=TravelerDetails(count=2),
                budget=BudgetDetails(),
                hard_constraints=HardConstraints(),
                soft_preferences=SoftPreferences(),
                contextual_signals=ContextualSignals(),
            ),
        )
        restored = SearchRequest.model_validate_json(req.model_dump_json())
        assert restored.workflow_id == "wf-1"
        assert restored.requirements.trip.destination == "Goa"

    def test_search_result_optional_fields(self) -> None:
        result = SearchResult(provider_ref="p1", label="Hotel A")
        assert result.price is None
        assert result.currency is None

    def test_search_response_empty_by_default(self) -> None:
        resp = SearchResponse(workflow_id="wf-1")
        assert resp.results == []

    def test_search_provider_is_runtime_checkable(self) -> None:
        assert isinstance(_FakeSearchProvider(), SearchProvider)


class _FakeSearchProvider:
    def search(self, request: SearchRequest) -> SearchResponse:
        return SearchResponse(workflow_id=request.workflow_id)


# ---------------------------------------------------------------------------
# Execution contracts
# ---------------------------------------------------------------------------


def _book_approval() -> Approval:
    return Approval(
        workflow_id="wf-1",
        action=ConsequentialAction.BOOK,
        decision=ApprovalDecision.APPROVED,
    )


class TestExecutionContracts:
    def test_execution_request_rejects_missing_approval(self) -> None:
        with pytest.raises(Exception):
            ExecutionRequest(workflow_id="wf-1", action=ConsequentialAction.BOOK, approval=None)

    def test_execution_request_rejects_rejected_approval(self) -> None:
        approval = Approval(
            workflow_id="wf-1",
            action=ConsequentialAction.BOOK,
            decision=ApprovalDecision.REJECTED,
        )
        with pytest.raises(Exception):
            ExecutionRequest(workflow_id="wf-1", action=ConsequentialAction.BOOK, approval=approval)

    def test_execution_request_accepts_approved_book(self) -> None:
        req = ExecutionRequest(
            workflow_id="wf-1",
            action=ConsequentialAction.BOOK,
            approval=_book_approval(),
        )
        assert req.workflow_id == "wf-1"

    def test_execution_result_round_trip(self) -> None:
        result = ExecutionResult(
            workflow_id="wf-1", provider_ref="r1", confirmed=True
        )
        restored = ExecutionResult.model_validate_json(result.model_dump_json())
        assert restored.confirmed is True

    def test_execution_provider_is_runtime_checkable(self) -> None:
        assert isinstance(_FakeExecutionProvider(), ExecutionProvider)


class _FakeExecutionProvider:
    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        return ExecutionResult(
            workflow_id=request.workflow_id,
            provider_ref="r1",
            confirmed=True,
        )


# ---------------------------------------------------------------------------
# Decision engine contracts
# ---------------------------------------------------------------------------


class TestDecisionContracts:
    def test_candidate_round_trip(self) -> None:
        c = Candidate(provider_ref="p1", label="Hotel A", price=Decimal("1000"))
        restored = Candidate.model_validate_json(c.model_dump_json())
        assert restored.price == Decimal("1000")

    def test_recommendation_input_round_trip(self) -> None:
        from app.models.travel_requirements import (
            BudgetDetails,
            ContextualSignals,
            HardConstraints,
            SoftPreferences,
            TravelerDetails,
            TravelRequirements,
            TripDetails,
        )

        inp = RecommendationInput(
            workflow_id="wf-1",
            requirements=TravelRequirements(
                trip=TripDetails(destination="Goa"),
                travelers=TravelerDetails(count=2),
                budget=BudgetDetails(),
                hard_constraints=HardConstraints(),
                soft_preferences=SoftPreferences(),
                contextual_signals=ContextualSignals(),
            ),
            candidates=[Candidate(provider_ref="c1", label="Option 1")],
        )
        restored = RecommendationInput.model_validate_json(inp.model_dump_json())
        assert len(restored.candidates) == 1
        assert restored.candidates[0].provider_ref == "c1"

    def test_recommendation_options_empty_by_default(self) -> None:
        rec = Recommendation(workflow_id="wf-1")
        assert rec.options == []

    def test_decision_engine_is_runtime_checkable(self) -> None:
        assert isinstance(_FakeDecisionEngine(), DecisionEngine)


class _FakeDecisionEngine:
    def recommend(self, input_: RecommendationInput) -> Recommendation:
        return Recommendation(workflow_id=input_.workflow_id)
