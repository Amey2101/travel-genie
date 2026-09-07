from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import pytest

from app.providers.llm import LLMExtractionOutput
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