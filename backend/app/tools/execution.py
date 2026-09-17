from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from app.agent.security import Approval, ConsequentialAction, assert_approved


class ExecutionRequest(BaseModel):
    """Contract for a consequential execution.

    Constructing an ``ExecutionRequest`` requires an APPROVED, matching
    ``Approval`` — the approval gate is enforced by the type system, not by
    caller discipline.
    """

    workflow_id: str
    action: ConsequentialAction
    approval: Approval
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _verify_approval(self) -> ExecutionRequest:
        try:
            assert_approved(self.approval, action=self.action)
        except Exception as exc:  # noqa: BLE001 - normalize to a validation error
            raise ValueError(str(exc)) from exc
        return self


class ExecutionResult(BaseModel):
    workflow_id: str
    provider_ref: str
    confirmed: bool


@runtime_checkable
class ExecutionProvider(Protocol):
    """External execution boundary.

    M4 defines the contract only; no provider implementation exists yet.
    """

    def execute(self, request: ExecutionRequest) -> ExecutionResult: ...