from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.actions import AgentResponse
from app.agent.errors import WorkflowNotFoundError
from app.agent.orchestrator import TravelAgent
from app.agent.state import AgentState


class CreateWorkflowResponse(BaseModel):
    workflow_id: str


class UserMessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


def workflows_router(agent: TravelAgent) -> APIRouter:
    """Build routes wired to the given agent instance (dependency injection)."""

    router = APIRouter(tags=["workflows"])

    @router.post("/workflows", response_model=CreateWorkflowResponse, status_code=201)
    def create_workflow() -> CreateWorkflowResponse:
        return CreateWorkflowResponse(workflow_id=agent.create_workflow())

    @router.post("/workflows/{workflow_id}/messages", response_model=AgentResponse)
    def send_message(workflow_id: str, body: UserMessageRequest) -> AgentResponse:
        try:
            return agent.handle_message(workflow_id, body.text)
        except WorkflowNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/workflows/{workflow_id}", response_model=AgentState)
    def get_workflow(workflow_id: str) -> AgentState:
        try:
            return agent.get_state(workflow_id)
        except WorkflowNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return router