from __future__ import annotations

from app.agent.state import AgentAction, AgentState, ClarificationState, WorkflowStatus
from app.models.travel_requirements import (
    ClarificationImportance,
    ClarificationItem,
)


class TestAgentState:
    def test_defaults(self) -> None:
        state = AgentState(workflow_id="wf-1")
        assert state.status is WorkflowStatus.COLLECTING_REQUIREMENTS
        assert state.conversation == []
        assert state.clarification == ClarificationState()
        assert state.current_action is None
        assert state.requirements is None
        assert state.errors == []

    def test_round_trip_serialisation(self) -> None:
        state = AgentState(workflow_id="wf-1")
        restored = AgentState.model_validate(state.model_dump())
        assert restored.workflow_id == "wf-1"
        assert restored.status is WorkflowStatus.COLLECTING_REQUIREMENTS

    def test_clarification_pending_round_trip(self) -> None:
        item = ClarificationItem(
            field="trip.origin",
            importance=ClarificationImportance.REQUIRED,
            question="Where will you be travelling from?",
        )
        state = AgentState(
            workflow_id="wf-1",
            clarification=ClarificationState(pending=[item]),
        )
        restored = AgentState.model_validate_json(state.model_dump_json())
        assert restored.clarification.pending[0].field == "trip.origin"
        assert restored.clarification.pending[0].importance is (
            ClarificationImportance.REQUIRED
        )

    def test_statuses_align_with_implemented_capabilities(self) -> None:
        statuses = {s.value for s in WorkflowStatus}
        assert statuses == {
            "collecting_requirements",
            "awaiting_clarification",
            "requirements_complete",
            "failed",
        }

    def test_actions_align_with_implemented_capabilities(self) -> None:
        actions = {a.value for a in AgentAction}
        assert actions == {
            "ask_clarification",
            "search",
            "report_error",
            "await_user",
        }