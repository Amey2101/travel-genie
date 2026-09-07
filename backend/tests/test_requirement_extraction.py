from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

from app.models.travel_requirements import (
    ClarificationImportance,
    ClarificationItem,
    ComfortLevel,
    DateFlexibility,
    DateRange,
    ExtractionMetadata,
    ExtractionResult,
    TravelerCategory,
    TravelRequirements,
    TransportMode,
)
from app.providers.llm import (
    BudgetExtraction,
    BudgetKind,
    CompanionExtraction,
    ContextualExtraction,
    LLMExtractionOutput,
    LLMExtractionRequest,
)
from app.providers.stub_llm import StubLLMClient
from app.services.requirement_extraction import (
    ExtractionError,
    InconsistentExtractionError,
    LLMOutputValidationError,
    RequirementExtractionService,
)


class TestFullPipeline:
    def test_complete_extraction(
        self, make_service, fixed_reference_datetime: datetime
    ) -> None:
        output = LLMExtractionOutput(
            trip_origin="Mumbai",
            trip_destination="Goa",
            trip_start_date=date(2026, 12, 20),
            trip_end_date=date(2026, 12, 24),
            duration_days=4,
            date_flexibility=DateFlexibility.EXACT,
            traveler_count=3,
            traveler_categories=[TravelerCategory.ADULT],
            companions=[CompanionExtraction(relationship="parents")],
            budget=BudgetExtraction(
                amount=Decimal("50000"), currency="INR", kind=BudgetKind.APPROXIMATE
            ),
            required_transport=TransportMode.FLIGHT,
            preferred_transport=TransportMode.TRAIN,
            comfort_level=ComfortLevel.STANDARD,
            contextual=ContextualExtraction(
                traveling_with_parents=True,
                has_children=False,
                first_time_destination=True,
                dietary_notes="vegetarian",
            ),
            confidence=0.88,
            notes=["window seats preferred"],
        )
        service = make_service(output)
        result = service.extract("Going to Goa with parents, budget around 50k")

        assert isinstance(result, ExtractionResult)

        req = result.requirements
        assert isinstance(req, TravelRequirements)
        assert req.trip.origin == "Mumbai"
        assert req.trip.destination == "Goa"
        assert req.trip.dates.date_range == DateRange(
            start=date(2026, 12, 20), end=date(2026, 12, 24)
        )
        assert req.trip.dates.duration_days == 4
        assert req.trip.dates.flexibility == DateFlexibility.EXACT
        assert req.travelers.count == 3
        assert req.travelers.categories == [TravelerCategory.ADULT]
        assert req.travelers.companions[0].relationship == "parents"
        assert req.budget.total == Decimal("50000")
        assert req.budget.is_approximate is True
        assert req.budget.per_category == {}
        assert req.hard_constraints.max_budget is None
        assert req.hard_constraints.required_transport == TransportMode.FLIGHT
        assert req.soft_preferences.preferred_transport == TransportMode.TRAIN
        assert req.soft_preferences.comfort_level == ComfortLevel.STANDARD
        assert req.contextual_signals.traveling_with_parents is True
        assert req.contextual_signals.accessibility_notes is None
        assert req.contextual_signals.dietary_notes == "vegetarian"

        assert result.clarification_needed == []

        meta = result.metadata
        assert meta.extraction_confidence == 0.88
        assert meta.notes == ["window seats preferred"]
        assert meta.original_user_message == (
            "Going to Goa with parents, budget around 50k"
        )
        assert meta.explicit_fields == [
            "trip.origin",
            "trip.destination",
            "trip.dates.date_range",
            "trip.dates.duration_days",
            "travelers.count",
            "budget.total",
        ]

    def test_metadata_is_in_extraction_result_not_travel_requirements(self, make_service):
        output = LLMExtractionOutput(trip_destination="Goa", confidence=0.6)
        result = make_service(output).extract("Goa")
        assert hasattr(result, "metadata")
        assert isinstance(result.metadata, ExtractionMetadata)
        assert not hasattr(result.requirements, "metadata")
        assert not hasattr(result.requirements, "extraction_confidence")

    def test_confidence_metadata_preserved(self, make_service):
        for conf in (0.0, 0.42, 1.0):
            output = LLMExtractionOutput(trip_destination="Goa", confidence=conf)
            result = make_service(output).extract("Goa")
            assert result.metadata.extraction_confidence == conf

    def test_confidence_none_preserved(self, make_service):
        output = LLMExtractionOutput(trip_destination="Goa", confidence=None)
        result = make_service(output).extract("Goa")
        assert result.metadata.extraction_confidence is None

    def test_stub_llmclient_drives_full_pipeline(self):
        output = LLMExtractionOutput(
            trip_origin="Delhi",
            trip_destination="Leh",
            duration_days=5,
            traveler_count=2,
        )
        stub = StubLLMClient(output)
        service = RequirementExtractionService(stub, reference_datetime=datetime(2026, 9, 3, 9, 0, tzinfo=timezone.utc))
        result = service.extract("Delhi to Leh, 5 days, 2 people")
        req = result.requirements
        assert req.trip.origin == "Delhi"
        assert req.trip.destination == "Leh"
        assert req.trip.dates.duration_days == 5
        assert req.travelers.count == 2

    def test_llm_output_validation_error(self, fixed_reference_datetime):
        bad_stub = StubLLMClient({"traveler_count": -1})
        service = RequirementExtractionService(
            bad_stub, reference_datetime=fixed_reference_datetime
        )
        try:
            result = service.extract("go somewhere")
            assert False, "Should have raised LLMOutputValidationError"
        except LLMOutputValidationError:
            pass
        except Exception as e:
            assert False, f"Expected LLMOutputValidationError, got {type(e).__name__}: {e}"

    def test_extraction_error_is_base_exception(self):
        assert issubclass(LLMOutputValidationError, ExtractionError)
        assert issubclass(ExtractionError, Exception)

    def test_service_independently_testable_with_stub(self, make_service):
        output = LLMExtractionOutput(trip_destination="Shimla", traveler_count=4)
        service = make_service(output)
        result = service.extract("Shimla with 4 friends")
        assert result.requirements.trip.destination == "Shimla"
        assert result.requirements.travelers.count == 4

    def test_parents_not_seniors_not_accessible(self, make_service):
        output = LLMExtractionOutput(
            trip_origin="Mumbai",
            trip_destination="Goa",
            duration_days=4,
            traveler_count=3,
            companions=[CompanionExtraction(relationship="parents")],
            contextual=ContextualExtraction(traveling_with_parents=True),
        )
        result = make_service(output).extract(
            "Taking my parents to Goa for 4 days"
        )
        req = result.requirements
        assert req.contextual_signals.traveling_with_parents is True
        assert req.travelers.companions[0].relationship == "parents"
        assert req.travelers.companions[0].categories == []
        assert TravelerCategory.SENIOR not in req.travelers.categories
        assert req.contextual_signals.accessibility_notes is None

    def test_prompt_keeps_parents_not_seniors_rule(self):
        stub = StubLLMClient(LLMExtractionOutput(trip_destination="Goa"))
        service = RequirementExtractionService(
            stub, reference_datetime=datetime(2026, 9, 3, tzinfo=timezone.utc)
        )
        service.extract("Taking my parents to Goa")
        prompt_lower = stub.last_request.system_prompt.lower()
        assert "not imply senior" in prompt_lower
        assert "mobility" in prompt_lower
        assert "accessibility" in prompt_lower

    def test_separation_of_stages(self, fixed_reference_datetime):
        stages_observed: list[str] = []

        class RecordingStubLLMClient:
            def generate_structured(
                self, *, request: LLMExtractionRequest, schema: type
            ):
                stages_observed.append("llm_call")
                return schema.model_validate(
                    LLMExtractionOutput(trip_destination="Goa")
                )

        service = RequirementExtractionService(
            RecordingStubLLMClient(), reference_datetime=fixed_reference_datetime
        )
        result = service.extract("Goa trip")
        assert stages_observed == ["llm_call"]
        assert result.requirements.trip.destination == "Goa"
        assert isinstance(result, ExtractionResult)

    def test_case_insensitive_clarification_questions(self, make_service):
        output = LLMExtractionOutput()
        result = make_service(output).extract("")
        for cl in result.clarification_needed:
            assert isinstance(cl, ClarificationItem)
            assert isinstance(cl.question, str)


class TestTemporalConsistency:
    def test_conflicting_range_and_duration_raises(self, make_service):
        output = LLMExtractionOutput(
            trip_origin="Mumbai",
            trip_destination="Goa",
            trip_start_date=date(2026, 12, 20),
            trip_end_date=date(2026, 12, 24),
            duration_days=10,
        )
        try:
            make_service(output).extract("Mumbai to Goa Dec 20-24 for 10 days")
            assert False, "Should have raised InconsistentExtractionError"
        except InconsistentExtractionError:
            pass

    def test_consistent_range_and_duration_ok(self, make_service):
        output = LLMExtractionOutput(
            trip_origin="Mumbai",
            trip_destination="Goa",
            trip_start_date=date(2026, 12, 20),
            trip_end_date=date(2026, 12, 24),
            duration_days=4,
        )
        result = make_service(output).extract("Mumbai to Goa Dec 20-24 for 4 days")
        req = result.requirements
        assert req.trip.dates.date_range == DateRange(
            start=date(2026, 12, 20), end=date(2026, 12, 24)
        )
        assert req.trip.dates.duration_days == 4

    def test_inconsistent_error_is_extraction_error(self):
        assert issubclass(InconsistentExtractionError, ExtractionError)


class TestClarificationDeduplication:
    def test_llm_clarification_duplicate_of_policy_field_dropped(self, make_service):
        from app.providers.llm import ClarificationExtraction

        output = LLMExtractionOutput(
            trip_destination="Goa",
            clarification_needed=[
                ClarificationExtraction(
                    field="trip.origin",
                    importance=ClarificationImportance.REQUIRED,
                    question="Different origin question?",
                    possible_interpretations=["from Pune?", "from Delhi?"],
                )
            ],
        )
        result = make_service(output).extract("Goa")
        origin_items = [cl for cl in result.clarification_needed if cl.field == "trip.origin"]
        assert len(origin_items) == 1
        assert origin_items[0].question == "Where will you be travelling from?"
        assert origin_items[0].possible_interpretations == ["from Pune?", "from Delhi?"]

    def test_non_policy_llm_clarification_preserved(self, make_service):
        from app.providers.llm import ClarificationExtraction

        output = LLMExtractionOutput(
            trip_origin="Mumbai",
            trip_destination="Goa",
            duration_days=3,
            traveler_count=2,
            clarification_needed=[
                ClarificationExtraction(
                    field="soft_preferences.comfort_level",
                    question="Do you prefer a specific comfort level?",
                )
            ],
        )
        result = make_service(output).extract("Mumbai to Goa")
        fields = [cl.field for cl in result.clarification_needed]
        assert "soft_preferences.comfort_level" in fields