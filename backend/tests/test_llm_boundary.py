from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from app.providers.llm import (
    LLMClient,
    LLMExtractionRequest,
    LLMExtractionOutput,
    BudgetExtraction,
    BudgetKind,
)
from app.providers.stub_llm import StubLLMClient
from app.models.travel_requirements import DateFlexibility

FIXED_DT = datetime(2026, 9, 3, 9, 0, tzinfo=timezone.utc)


def _sample_output() -> LLMExtractionOutput:
    return LLMExtractionOutput(
        trip_origin="Mumbai",
        trip_destination="Goa",
        trip_start_date=date(2026, 12, 20),
        trip_end_date=date(2026, 12, 24),
        date_flexibility=DateFlexibility.EXACT,
        traveler_count=2,
        budget=BudgetExtraction(
            amount="50000",
            currency="INR",
            kind=BudgetKind.APPROXIMATE,
        ),
    )


class TestLLMClientProtocol:
    def test_stub_is_an_llm_client(self) -> None:
        stub = StubLLMClient(_sample_output())
        assert isinstance(stub, LLMClient)

    def test_generic_generate_structured_returns_requested_type(self) -> None:
        stub = StubLLMClient(_sample_output())
        request = LLMExtractionRequest(
            system_prompt="sys",
            user_message="hello",
            reference_datetime=FIXED_DT,
        )
        result = stub.generate_structured(request=request, schema=LLMExtractionOutput)
        assert isinstance(result, LLMExtractionOutput)
        assert result.trip_destination == "Goa"

    def test_stub_records_last_request(self) -> None:
        stub = StubLLMClient(_sample_output())
        request = LLMExtractionRequest(
            system_prompt="sys-prompt",
            user_message="book a trip",
            reference_datetime=FIXED_DT,
        )
        stub.generate_structured(request=request, schema=LLMExtractionOutput)
        assert stub.last_request is not None
        assert stub.last_request.user_message == "book a trip"
        assert stub.last_request.system_prompt == "sys-prompt"

    def test_stub_validates_configured_output(self) -> None:
        """Malformed configured output is rejected during validation."""
        stub = StubLLMClient({"traveler_count": -1})
        request = LLMExtractionRequest(
            system_prompt="s",
            user_message="m",
            reference_datetime=FIXED_DT,
        )
        with pytest.raises(ValidationError):
            stub.generate_structured(request=request, schema=LLMExtractionOutput)

    def test_stub_factory_invoked_with_request(self) -> None:
        captured: list[LLMExtractionRequest] = []

        def factory(request: LLMExtractionRequest) -> LLMExtractionOutput:
            captured.append(request)
            return LLMExtractionOutput(trip_destination="Goa")

        stub = StubLLMClient(factory=factory)
        request = LLMExtractionRequest(
            system_prompt="s",
            user_message="to Goa",
            reference_datetime=FIXED_DT,
        )
        result = stub.generate_structured(request=request, schema=LLMExtractionOutput)
        assert result.trip_destination == "Goa"
        assert len(captured) == 1
        assert captured[0].user_message == "to Goa"


class TestLLMExtractionRequest:
    def test_request_carries_reference_datetime(self, fixed_reference_datetime) -> None:
        request = LLMExtractionRequest(
            system_prompt="s",
            user_message="m",
            reference_datetime=fixed_reference_datetime,
        )
        assert request.reference_datetime == fixed_reference_datetime

    def test_default_schema_rejection_for_negative_duration(self) -> None:
        with pytest.raises(ValidationError):
            LLMExtractionOutput(duration_days=0)

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            LLMExtractionOutput(confidence=1.5)
        out = LLMExtractionOutput(confidence=0.8)
        assert out.confidence == 0.8
