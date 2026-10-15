from __future__ import annotations

from pydantic import BaseModel, Field

from app.agent.state import AgentAction, WorkflowError, WorkflowStatus
from app.decision.budget import BudgetEstimate
from app.decision.engine import Recommendation
from app.models.travel_requirements import ClarificationItem, TravelRequirements
from app.tools.search import SearchRequest


class AgentResponse(BaseModel):
    """Structured outcome of one agent turn."""

    workflow_id: str
    status: WorkflowStatus
    action: AgentAction
    reply: str
    clarification_questions: list[ClarificationItem] = Field(default_factory=list)
    requirements: TravelRequirements | None = None
    search_handoff: SearchRequest | None = None
    recommendation: Recommendation | None = None
    budget_estimate: BudgetEstimate | None = None
    error: WorkflowError | None = None