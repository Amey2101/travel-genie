from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from app.decision.budget import BudgetEstimate
from app.decision.engine import Recommendation
from app.models.travel_requirements import (
    ClarificationItem,
    ExtractionResult,
    TravelRequirements,
)
from app.tools.search import SearchResult


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class WorkflowStatus(str, Enum):
    """Lifecycle statuses reachable with the currently implemented capabilities.

    Downstream stages (search, decision, approval, execution) are not reachable
    yet, so they are deliberately absent from this enum.
    """

    COLLECTING_REQUIREMENTS = "collecting_requirements"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    REQUIREMENTS_COMPLETE = "requirements_complete"
    FAILED = "failed"


class AgentAction(str, Enum):
    """Deterministic next step the agent chooses as the outcome of a turn."""

    ASK_CLARIFICATION = "ask_clarification"
    SEARCH = "search"
    REPORT_ERROR = "report_error"
    AWAIT_USER = "await_user"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class WorkflowMessage(BaseModel):
    role: MessageRole
    text: str
    timestamp: datetime = Field(default_factory=_utc_now)


class ClarificationState(BaseModel):
    """Currently open clarification items.

    ``pending`` is replaced wholesale after every re-extraction. The answer
    history lives in ``AgentState.conversation``; fields are not mapped to
    answers here.
    """

    pending: list[ClarificationItem] = Field(default_factory=list)


class WorkflowError(BaseModel):
    """Structured, non-fabricated failure recorded in agent state."""

    code: str
    message: str
    occurred_at: datetime = Field(default_factory=_utc_now)


class AgentState(BaseModel):
    """Typed, serializable state for one travel workflow."""

    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.COLLECTING_REQUIREMENTS
    conversation: list[WorkflowMessage] = Field(default_factory=list)
    requirements: TravelRequirements | None = None
    last_extraction: ExtractionResult | None = None
    clarification: ClarificationState = Field(default_factory=ClarificationState)
    current_action: AgentAction | None = None
    search_results: list[SearchResult] = Field(default_factory=list)
    recommendation: Recommendation | None = None
    budget_estimate: BudgetEstimate | None = None
    errors: list[WorkflowError] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utc_now)