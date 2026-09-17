from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agent.security import (
    Approval,
    ApprovalDecision,
    ConsequentialAction,
    ConsequentialActionWithoutApprovalError,
    assert_approved,
    requires_approval,
)
from app.tools.execution import ExecutionRequest


def _approved(workflow_id: str = "w", action: ConsequentialAction = ConsequentialAction.BOOK) -> Approval:
    return Approval(
        workflow_id=workflow_id,
        action=action,
        decision=ApprovalDecision.APPROVED,
    )


def _rejected(workflow_id: str = "w", action: ConsequentialAction = ConsequentialAction.BOOK) -> Approval:
    return Approval(
        workflow_id=workflow_id,
        action=action,
        decision=ApprovalDecision.REJECTED,
    )


class TestRequiresApproval:
    def test_all_consequential_actions_require_approval(self) -> None:
        for action in ConsequentialAction:
            assert requires_approval(action) is True


class TestAssertApproved:
    def test_none_raises(self) -> None:
        try:
            assert_approved(None, action=ConsequentialAction.BOOK)
            assert False, "expected ConsequentialActionWithoutApprovalError"
        except ConsequentialActionWithoutApprovalError:
            pass

    def test_rejected_raises(self) -> None:
        try:
            assert_approved(_rejected(), action=ConsequentialAction.BOOK)
            assert False, "expected ConsequentialActionWithoutApprovalError"
        except ConsequentialActionWithoutApprovalError:
            pass

    def test_action_mismatch_raises(self) -> None:
        approval = _approved(action=ConsequentialAction.CANCEL)
        try:
            assert_approved(approval, action=ConsequentialAction.BOOK)
            assert False, "expected ConsequentialActionWithoutApprovalError"
        except ConsequentialActionWithoutApprovalError:
            pass

    def test_matching_approved_passes(self) -> None:
        assert_approved(_approved(), action=ConsequentialAction.BOOK)

    def test_error_is_agent_error(self) -> None:
        from app.agent.errors import AgentError
        assert issubclass(ConsequentialActionWithoutApprovalError, AgentError)


class TestApprovalSourceLock:
    def test_source_must_be_user(self) -> None:
        try:
            Approval(
                workflow_id="w",
                action=ConsequentialAction.BOOK,
                decision=ApprovalDecision.APPROVED,
                source="system",
            )
            assert False, "expected ValidationError"
        except ValidationError:
            pass

    def test_source_default_is_user(self) -> None:
        approval = Approval(
            workflow_id="w",
            action=ConsequentialAction.BOOK,
            decision=ApprovalDecision.APPROVED,
        )
        assert approval.source == "user"


class TestExecutionRequestApprovalGate:
    def test_rejected_approval_rejected_at_construction(self) -> None:
        try:
            ExecutionRequest(
                workflow_id="w",
                action=ConsequentialAction.BOOK,
                approval=_rejected(),
            )
            assert False, "expected ValidationError"
        except ValidationError:
            pass

    def test_wrong_action_in_approval_rejected(self) -> None:
        approval = _approved(action=ConsequentialAction.CANCEL)
        try:
            ExecutionRequest(
                workflow_id="w",
                action=ConsequentialAction.BOOK,
                approval=approval,
            )
            assert False, "expected ValidationError"
        except ValidationError:
            pass

    def test_approved_matching_action_succeeds(self) -> None:
        req = ExecutionRequest(
            workflow_id="w",
            action=ConsequentialAction.BOOK,
            approval=_approved(),
        )
        assert req.workflow_id == "w"
        assert req.action is ConsequentialAction.BOOK