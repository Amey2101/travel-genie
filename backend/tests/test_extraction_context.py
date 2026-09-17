from __future__ import annotations

from datetime import datetime, timezone

from app.providers.llm import LLMExtractionOutput, LLMExtractionRequest
from app.providers.stub_llm import StubLLMClient
from app.services.requirement_extraction import RequirementExtractionService

FIXED_DT = datetime(2026, 9, 3, 9, 0, tzinfo=timezone.utc)


def _service_with_capturing_stub() -> tuple[
    RequirementExtractionService, list[LLMExtractionRequest]
]:
    calls: list[LLMExtractionRequest] = []

    def _factory(request: LLMExtractionRequest) -> LLMExtractionOutput:
        calls.append(request)
        return LLMExtractionOutput(trip_destination="Goa")

    client = StubLLMClient(factory=_factory)
    service = RequirementExtractionService(client, reference_datetime=FIXED_DT)
    return service, calls


class TestContextExtraction:
    def test_context_prepended_to_llm_request(self) -> None:
        service, calls = _service_with_capturing_stub()
        service.extract(
            "from Mumbai, 3 days",
            context=["Goa trip"],
        )
        assert len(calls) == 1
        assert calls[0].user_message == "Goa trip\nfrom Mumbai, 3 days"

    def test_no_context_backward_compat(self) -> None:
        service, calls = _service_with_capturing_stub()
        service.extract("solo message")
        assert calls[0].user_message == "solo message"

    def test_empty_context_list_ignored(self) -> None:
        service, calls = _service_with_capturing_stub()
        service.extract("solo", context=[])
        assert calls[0].user_message == "solo"

    def test_metadata_keeps_latest_message(self) -> None:
        service, _calls = _service_with_capturing_stub()
        result = service.extract("answer", context=["prior"])
        assert result.metadata.original_user_message == "answer"

    def test_three_turn_context(self) -> None:
        service, calls = _service_with_capturing_stub()
        service.extract(
            "third turn",
            context=["first turn", "second turn"],
        )
        assert calls[0].user_message == (
            "first turn\nsecond turn\nthird turn"
        )
