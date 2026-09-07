from __future__ import annotations

from datetime import date, datetime, timezone

from app.providers.llm import LLMExtractionOutput, LLMExtractionRequest
from app.providers.stub_llm import StubLLMClient
from app.services.requirement_extraction import RequirementExtractionService


class TestReferenceDatetime:
    def test_reference_datetime_is_injectable(
        self, fixed_reference_datetime: datetime
    ) -> None:
        stub = StubLLMClient(LLMExtractionOutput(trip_destination="Goa"))
        service = RequirementExtractionService(
            stub, reference_datetime=fixed_reference_datetime
        )
        service.extract("Goa trip")
        assert stub.last_request is not None
        assert stub.last_request.reference_datetime == fixed_reference_datetime

    def test_request_receives_correct_reference_datetime(self, fixed_reference_datetime):
        captured_requests: list[LLMExtractionRequest] = []

        def factory(request: LLMExtractionRequest) -> LLMExtractionOutput:
            captured_requests.append(request)
            return LLMExtractionOutput(trip_destination="Goa")

        stub = StubLLMClient(factory=factory)
        service = RequirementExtractionService(
            stub, reference_datetime=fixed_reference_datetime
        )
        service.extract("Goa trip")
        assert len(captured_requests) == 1
        assert captured_requests[0].reference_datetime == fixed_reference_datetime

    def test_prompt_contains_correct_reference_date(self, fixed_reference_datetime):
        captured_requests: list[LLMExtractionRequest] = []

        def factory(request: LLMExtractionRequest) -> LLMExtractionOutput:
            captured_requests.append(request)
            return LLMExtractionOutput(trip_destination="Goa")

        stub = StubLLMClient(factory=factory)
        service = RequirementExtractionService(
            stub, reference_datetime=fixed_reference_datetime
        )
        service.extract("Goa trip")
        prompt = captured_requests[0].system_prompt
        assert "2026-09-03" in prompt

    def test_service_accepts_absolute_dates_from_llm(self, make_service):
        output = LLMExtractionOutput(
            trip_destination="Goa",
            trip_start_date=date(2026, 12, 20),
            trip_end_date=date(2026, 12, 24),
        )
        req = make_service(output).extract("Goa Dec 20-24").requirements
        assert req.trip.dates.date_range.start == date(2026, 12, 20)
        assert req.trip.dates.date_range.end == date(2026, 12, 24)

    def test_missing_dates_no_temporal_interpretation(self, make_service):
        output = LLMExtractionOutput(trip_destination="Goa")
        req = make_service(output).extract("Goa trip").requirements
        assert req.trip.dates is None

    def test_default_reference_datetime_when_none(self):
        """When no reference_datetime is injected, the service uses datetime.now(UTC)."""
        stub = StubLLMClient(LLMExtractionOutput(trip_destination="Goa"))
        service = RequirementExtractionService(stub, reference_datetime=None)
        before = datetime.now(timezone.utc)
        service.extract("Goa trip")
        after = datetime.now(timezone.utc)
        assert stub.last_request is not None
        ref = stub.last_request.reference_datetime
        assert before <= ref <= after