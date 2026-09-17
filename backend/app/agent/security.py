from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from app.agent.errors import AgentError


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ConsequentialAction(str, Enum):
    """Action categories that require explicit user approval before execution."""

    BOOK = "book"
    CANCEL = "cancel"
    PAYMENT = "payment"


class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class Approval(BaseModel):
    """A recorded, explicit user decision.

    ``source`` is locked to "user": an approval can never originate from the
    LLM or the system.
    """

    workflow_id: str
    action: ConsequentialAction
    decision: ApprovalDecision
    source: Literal["user"] = "user"
    decided_at: datetime = Field(default_factory=_utc_now)


class ConsequentialActionWithoutApprovalError(AgentError):
    """Raised if a consequential action is attempted without an explicit approval."""


def requires_approval(action: ConsequentialAction) -> bool:
    """M4 policy: every consequential action requires explicit user approval."""
    return True


def assert_approved(approval: Approval | None, *, action: ConsequentialAction) -> None:
    """Enforce the approval gate.

    Only a non-null, matching, APPROVED approval passes. An LLM response can
    never satisfy this guard.
    """
    if (
        approval is None
        or approval.action != action
        or approval.decision != ApprovalDecision.APPROVED
    ):
        raise ConsequentialActionWithoutApprovalError(
            f"consequential action {action.value!r} requires explicit user approval"
        )