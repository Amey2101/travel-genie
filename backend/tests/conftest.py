from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import pytest

from app.agent.orchestrator import TravelAgent
from app.agent.store import InMemoryWorkflowStore
from app.providers.llm import LLMExtractionOutput, LLMExtractionRequest
from app.providers.stub_llm import StubLLMClient
from app.services.requirement_extraction import RequirementExtractionService


@pytest.fixture
def fixed_reference_datetime() -> datetime:
    """A known reference time for deterministic temporal tests."""
    return datetime(2026, 9, 3, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def make_service(
    fixed_reference_datetime: datetime,
) -> Callable[..., RequirementExtractionService]:
    """Factory for a RequirementExtractionService wired to a fresh StubLLMClient."""

    def _make(output: LLMExtractionOutput) -> RequirementExtractionService:
        client = StubLLMClient(output)
        return RequirementExtractionService(
            client, reference_datetime=fixed_reference_datetime
        )

    return _make


@pytest.fixture
def make_agent(
    fixed_reference_datetime: datetime,
) -> Callable[..., tuple[TravelAgent, list[LLMExtractionRequest]]]:
    """Build a TravelAgent wired to a scripted sequence of LLM outputs.

    Returns ``(agent, calls)`` where ``calls`` is a list of
    ``LLMExtractionRequest`` objects captured after each LLM invocation.
    """

    def _make(
        script: list[LLMExtractionOutput],
    ) -> tuple[TravelAgent, list[LLMExtractionRequest]]:
        return _make_agent_from_script(script, fixed_reference_datetime)

    return _make


def _make_agent_from_script(
    script: list[LLMExtractionOutput],
    reference_datetime: datetime | None = None,
) -> tuple[TravelAgent, list[LLMExtractionRequest]]:
    """Standalone factory usable outside pytest fixture context."""
    dt = reference_datetime or datetime.now(timezone.utc)
    calls: list[LLMExtractionRequest] = []

    def _factory(request: LLMExtractionRequest) -> LLMExtractionOutput:
        idx = len(calls)
        out = script[idx] if idx < len(script) else LLMExtractionOutput()
        calls.append(request)
        return out

    client = StubLLMClient(factory=_factory)
    extraction = RequirementExtractionService(client, reference_datetime=dt)
    agent = TravelAgent(extraction=extraction, store=InMemoryWorkflowStore())
    return agent, calls