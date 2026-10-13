from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.actions import AgentResponse
from app.agent.errors import WorkflowNotFoundError
from app.agent.orchestrator import TravelAgent
from app.agent.state import AgentState
from app.services.requirement_extraction import ExtractionError
from app.services.structured_requirements import (
    ClarificationAnswer,
    ClarificationError,
    StructuredRequirementsRequest,
    build_extraction_result,
    merge_clarification,
)


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

    @router.post("/workflows/{workflow_id}/requirements", response_model=AgentResponse)
    def send_requirements(
        workflow_id: str, body: StructuredRequirementsRequest
    ) -> AgentResponse:
        try:
            result = build_extraction_result(body)
        except ExtractionError as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    "The trip details are internally inconsistent and could not "
                    "be processed. Please check the dates and duration."
                ),
            ) from exc
        try:
            return agent.handle_structured(workflow_id, result)
        except WorkflowNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/workflows/{workflow_id}/clarifications", response_model=AgentResponse)
    def answer_clarification(
        workflow_id: str, body: ClarificationAnswer
    ) -> AgentResponse:
        """Deterministically answer open clarifications (no LLM required).

        Reuses the current extraction, merges the supplied values, and resumes
        the same clarification/search/decision pipeline via ``handle_structured``.
        """
        try:
            state = agent.get_state(workflow_id)
        except WorkflowNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if state.last_extraction is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "There is nothing to clarify yet. Submit trip details before "
                    "answering clarification questions."
                ),
            )
        try:
            merged = merge_clarification(state.last_extraction, body)
        except ClarificationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            return agent.handle_structured(workflow_id, merged)
        except WorkflowNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/workflows/{workflow_id}", response_model=AgentState)
    def get_workflow(workflow_id: str) -> AgentState:
        try:
            return agent.get_state(workflow_id)
        except WorkflowNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return router