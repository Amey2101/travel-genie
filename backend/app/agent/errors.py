from __future__ import annotations


class AgentError(Exception):
    """Base error for the agent workflow layer."""


class WorkflowNotFoundError(AgentError):
    """Raised when an unknown workflow id is addressed."""

    def __init__(self, workflow_id: str) -> None:
        super().__init__(f"workflow not found: {workflow_id}")
        self.workflow_id = workflow_id