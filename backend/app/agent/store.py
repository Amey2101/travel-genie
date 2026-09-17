from __future__ import annotations

from typing import Protocol

from app.agent.state import AgentState


class WorkflowStore(Protocol):
    """Persistence boundary for workflow state (in-memory in M4, MongoDB later)."""

    def create(self, state: AgentState) -> None: ...

    def get(self, workflow_id: str) -> AgentState | None: ...

    def save(self, state: AgentState) -> None: ...


class InMemoryWorkflowStore:
    """Process-local store, sufficient while the app runs single-process."""

    def __init__(self) -> None:
        self._states: dict[str, AgentState] = {}

    def create(self, state: AgentState) -> None:
        self._states[state.workflow_id] = state

    def get(self, workflow_id: str) -> AgentState | None:
        return self._states.get(workflow_id)

    def save(self, state: AgentState) -> None:
        self._states[state.workflow_id] = state