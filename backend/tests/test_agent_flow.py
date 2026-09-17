from __future__ import annotations

from app.agent.errors import WorkflowNotFoundError
from app.agent.state import AgentAction, AgentState, MessageRole, WorkflowStatus
from app.models.travel_requirements import (
    BudgetDetails,
    ClarificationImportance,
    ContextualSignals,
    ExtractionMetadata,
    ExtractionResult,
    HardConstraints,
    SoftPreferences,
    TravelerDetails,
    TravelRequirements,
    TripDetails,
)
from app.providers.llm import LLMExtractionOutput
from app.providers.stub_llm import StubLLMClient
from app.services.requirement_extraction import RequirementExtractionService


def _goa_only() -> LLMExtractionOutput:
    return LLMExtractionOutput(trip_destination="Goa")


def _complete() -> LLMExtractionOutput:
    return LLMExtractionOutput(
        trip_origin="Mumbai",
        trip_destination="Goa",
        duration_days=3,
        traveler_count=2,
    )


def _bangalore_complete() -> LLMExtractionOutput:
    return LLMExtractionOutput(
        trip_origin="Mumbai",
        trip_destination="Bangalore",
        duration_days=3,
        traveler_count=2,
    )


def _failure_agent() -> tuple[RequirementExtractionService, StubLLMClient]:
    client = StubLLMClient({"traveler_count": -1})
    service = RequirementExtractionService(client)
    return service, client


class TestWorkflowLifecycle:
    def test_create_workflow(self, make_agent) -> None:
        agent, _calls = make_agent([])
        workflow_id = agent.create_workflow()
        assert isinstance(workflow_id, str)
        state = agent.get_state(workflow_id)
        assert state.status is WorkflowStatus.COLLECTING_REQUIREMENTS

    def test_get_state_unknown_workflow_raises(self, make_agent) -> None:
        agent, _calls = make_agent([])
        try:
            agent.get_state("nope")
            assert False, "expected WorkflowNotFoundError"
        except WorkflowNotFoundError:
            pass

    def test_handle_message_unknown_workflow_raises(self, make_agent) -> None:
        agent, _calls = make_agent([])
        try:
            agent.handle_message("nope", "hello")
            assert False, "expected WorkflowNotFoundError"
        except WorkflowNotFoundError:
            pass

    def test_complete_request_reaches_search_handoff(self, make_agent) -> None:
        agent, calls = make_agent([_complete()])
        workflow_id = agent.create_workflow()
        response = agent.handle_message(
            workflow_id, "Mumbai to Goa, 3 days, 2 people"
        )
        assert response.status is WorkflowStatus.REQUIREMENTS_COMPLETE
        assert response.action is AgentAction.SEARCH
        assert response.clarification_questions == []
        assert response.error is None
        assert response.search_handoff is not None
        assert response.search_handoff.workflow_id == workflow_id
        assert response.search_handoff.requirements.trip.destination == "Goa"
        assert response.requirements is not None
        assert len(calls) == 1

    def test_complete_reply_summarises_requirements(self, make_agent) -> None:
        agent, _calls = make_agent([_complete()])
        workflow_id = agent.create_workflow()
        response = agent.handle_message(
            workflow_id, "Mumbai to Goa, 3 days, 2 people"
        )
        assert "Requirements are complete" in response.reply
        assert "Mumbai to Goa" in response.reply
        assert "Next step: search" in response.reply

    def test_missing_required_information_asks_clarification(self, make_agent) -> None:
        agent, _calls = make_agent([_goa_only()])
        workflow_id = agent.create_workflow()
        response = agent.handle_message(workflow_id, "Goa trip")
        assert response.status is WorkflowStatus.AWAITING_CLARIFICATION
        assert response.action is AgentAction.ASK_CLARIFICATION
        fields = {item.field for item in response.clarification_questions}
        assert "trip.origin" in fields
        assert "travel.timing" in fields
        assert response.search_handoff is None
        state = agent.get_state(workflow_id)
        assert state.status is WorkflowStatus.AWAITING_CLARIFICATION
        pending_fields = {item.field for item in state.clarification.pending}
        assert pending_fields == {"trip.origin", "travel.timing"}

    def test_recommended_questions_do_not_block(self, make_agent) -> None:
        agent, _calls = make_agent([_complete()])
        workflow_id = agent.create_workflow()
        response = agent.handle_message(
            workflow_id, "Mumbai to Goa, 3 days, 2 people"
        )
        assert response.status is WorkflowStatus.REQUIREMENTS_COMPLETE

    def test_clarification_followed_by_answer(self, make_agent) -> None:
        agent, calls = make_agent([_goa_only(), _complete()])
        workflow_id = agent.create_workflow()
        first = agent.handle_message(workflow_id, "Goa trip")
        assert first.status is WorkflowStatus.AWAITING_CLARIFICATION

        second = agent.handle_message(
            workflow_id, "from Mumbai, 3 days, 2 people"
        )
        assert second.status is WorkflowStatus.REQUIREMENTS_COMPLETE
        assert second.clarification_questions == []
        assert len(calls) == 2
        assert "Goa trip" in calls[1].user_message
        assert "from Mumbai" in calls[1].user_message

    def test_update_after_complete_changes_requirements(self, make_agent) -> None:
        agent, _calls = make_agent([_complete(), _bangalore_complete()])
        workflow_id = agent.create_workflow()
        agent.handle_message(workflow_id, "Mumbai to Goa, 3 days, 2 people")
        updated = agent.handle_message(
            workflow_id, "actually Bangalore instead"
        )
        assert updated.status is WorkflowStatus.REQUIREMENTS_COMPLETE
        assert updated.requirements.trip.destination == "Bangalore"

    def test_update_can_regress_to_clarification(self, make_agent) -> None:
        agent, _calls = make_agent([_complete(), _goa_only()])
        workflow_id = agent.create_workflow()
        agent.handle_message(workflow_id, "Mumbai to Goa, 3 days, 2 people")
        regressed = agent.handle_message(workflow_id, "changed my mind")
        assert regressed.status is WorkflowStatus.AWAITING_CLARIFICATION
        assert regressed.action is AgentAction.ASK_CLARIFICATION

    def test_conversation_log_records_both_roles(self, make_agent) -> None:
        agent, _calls = make_agent([_complete()])
        workflow_id = agent.create_workflow()
        response = agent.handle_message(
            workflow_id, "Mumbai to Goa, 3 days, 2 people"
        )
        state: AgentState = agent.get_state(workflow_id)
        assert [m.role for m in state.conversation] == [
            MessageRole.USER,
            MessageRole.ASSISTANT,
        ]
        assert state.conversation[0].text == "Mumbai to Goa, 3 days, 2 people"
        assert state.conversation[1].text == response.reply


class TestFailureStates:
    def test_invalid_extraction_moves_to_failed(self, make_agent) -> None:
        agent, _calls = make_agent([_complete()])
        workflow_id = agent.create_workflow()
        service, client = _failure_agent()
        agent._extraction = service  # substitute a failing extraction path
        response = agent.handle_message(workflow_id, "go somewhere")
        assert response.status is WorkflowStatus.FAILED
        assert response.action is AgentAction.REPORT_ERROR
        assert response.requirements is None
        assert response.error is not None
        assert response.error.code == "llm_invalid_output"
        assert "couldn't understand" in response.reply
        state = agent.get_state(workflow_id)
        assert len(state.errors) == 1
        assert state.errors[0].code == "llm_invalid_output"
        assert state.last_extraction is None
        assert client.last_request is not None

    def test_failed_status_is_terminal(self, make_agent) -> None:
        agent, _calls = make_agent([])
        service, client = _failure_agent()
        agent = type(agent)(
            extraction=service, store=agent._store
        )
        workflow_id = agent.create_workflow()
        agent.handle_message(workflow_id, "first message")
        response = agent.handle_message(workflow_id, "try again")
        assert response.status is WorkflowStatus.FAILED
        assert response.action is AgentAction.AWAIT_USER
        assert response.error.code == "llm_invalid_output"
        assert "ended with an error" in response.reply
        assert client.last_request is not None
        assert client.last_request.user_message == "first message"
        state = agent.get_state(workflow_id)
        assert len(state.errors) == 1

    def test_failure_reply_never_fabricates_success(self, make_agent) -> None:
        agent, _calls = make_agent([_complete()])
        workflow_id = agent.create_workflow()
        service, _client = _failure_agent()
        agent._extraction = service
        response = agent.handle_message(workflow_id, "go somewhere")
        assert response.search_handoff is None
        assert response.status is WorkflowStatus.FAILED
        assert "Requirements are complete" not in response.reply