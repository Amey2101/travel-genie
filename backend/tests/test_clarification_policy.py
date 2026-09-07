from __future__ import annotations

from datetime import date

from app.models.travel_requirements import ClarificationImportance
from app.providers.llm import LLMExtractionOutput, ContextualExtraction


class TestClarificationPolicy:
    def test_missing_origin_is_required(self, make_service):
        output = LLMExtractionOutput(
            trip_destination="Goa",
            duration_days=3,
            traveler_count=2,
        )
        result = make_service(output).extract("Goa trip")
        fields = {cl.field for cl in result.clarification_needed}
        assert "trip.origin" in fields
        assert result.clarification_needed[
            next(i for i, c in enumerate(result.clarification_needed) if c.field == "trip.origin")
        ].importance == ClarificationImportance.REQUIRED

    def test_missing_destination_is_required(self, make_service):
        output = LLMExtractionOutput(
            trip_origin="Mumbai",
            duration_days=3,
            traveler_count=2,
        )
        result = make_service(output).extract("from Mumbai")
        fields = {cl.field for cl in result.clarification_needed}
        assert "trip.destination" in fields
        assert result.clarification_needed[
            next(i for i, c in enumerate(result.clarification_needed) if c.field == "trip.destination")
        ].importance == ClarificationImportance.REQUIRED

    def test_missing_timing_is_required(self, make_service):
        output = LLMExtractionOutput(
            trip_origin="Mumbai",
            trip_destination="Goa",
            traveler_count=2,
        )
        result = make_service(output).extract("Mumbai to Goa")
        fields = {cl.field for cl in result.clarification_needed}
        assert "travel.timing" in fields
        assert result.clarification_needed[
            next(i for i, c in enumerate(result.clarification_needed) if c.field == "travel.timing")
        ].importance == ClarificationImportance.REQUIRED

    def test_missing_budget_is_recommended(self, make_service):
        output = LLMExtractionOutput(
            trip_origin="Mumbai",
            trip_destination="Goa",
            duration_days=3,
            traveler_count=2,
        )
        result = make_service(output).extract("Mumbai to Goa, 3 days, 2 people")
        fields = {cl.field for cl in result.clarification_needed}
        assert "budget.total" in fields
        assert result.clarification_needed[
            next(i for i, c in enumerate(result.clarification_needed) if c.field == "budget.total")
        ].importance == ClarificationImportance.RECOMMENDED

    def test_missing_traveler_count_is_recommended(self, make_service):
        output = LLMExtractionOutput(
            trip_origin="Mumbai",
            trip_destination="Goa",
            duration_days=3,
        )
        result = make_service(output).extract("Mumbai to Goa, 3 days")
        fields = {cl.field for cl in result.clarification_needed}
        assert "travelers.count" in fields
        assert result.clarification_needed[
            next(i for i, c in enumerate(result.clarification_needed) if c.field == "travelers.count")
        ].importance == ClarificationImportance.RECOMMENDED

    def test_no_clarification_for_optional_signals_when_absent(self, make_service):
        output = LLMExtractionOutput(
            trip_origin="Mumbai",
            trip_destination="Goa",
            duration_days=3,
            traveler_count=2,
        )
        result = make_service(output).extract("Mumbai to Goa")
        generated_fields = {cl.field for cl in result.clarification_needed}
        assert "contextual_signals" not in generated_fields
        assert "contextual_signals.accessibility" not in generated_fields
        assert "contextual_signals.dietary" not in generated_fields

    def test_parents_no_senior_no_accessibility(self, make_service):
        from app.providers.llm import CompanionExtraction, ContextualExtraction
        output = LLMExtractionOutput(
            trip_origin="Mumbai",
            trip_destination="Goa",
            duration_days=3,
            traveler_count=3,
            companions=[CompanionExtraction(relationship="parents")],
            contextual=ContextualExtraction(
                traveling_with_parents=True,
            ),
        )
        result = make_service(output).extract("Taking my parents to Goa")
        req = result.requirements
        assert req.contextual_signals.traveling_with_parents is True
        assert req.travelers.companions[0].relationship == "parents"
        assert req.travelers.companions[0].categories == []
        assert req.travelers.categories == []
        assert req.contextual_signals.accessibility_notes is None

    def test_empty_output_all_required_and_recommended(self, make_service):
        output = LLMExtractionOutput()
        result = make_service(output).extract("")
        importance_map = {cl.field: cl.importance for cl in result.clarification_needed}
        assert importance_map.get("trip.origin") == ClarificationImportance.REQUIRED
        assert importance_map.get("trip.destination") == ClarificationImportance.REQUIRED
        assert importance_map.get("travel.timing") == ClarificationImportance.REQUIRED
        assert importance_map.get("budget.total") == ClarificationImportance.RECOMMENDED
        assert importance_map.get("travelers.count") == ClarificationImportance.RECOMMENDED

    def test_start_date_only_is_incomplete_timing(self, make_service):
        output = LLMExtractionOutput(
            trip_origin="Mumbai",
            trip_destination="Goa",
            trip_start_date=date(2026, 12, 20),
            traveler_count=2,
        )
        result = make_service(output).extract("Mumbai to Goa starting Dec 20")
        fields = {cl.field for cl in result.clarification_needed}
        assert "travel.timing" in fields
        assert result.requirements.trip.dates is None

    def test_end_date_only_is_incomplete_timing(self, make_service):
        output = LLMExtractionOutput(
            trip_origin="Mumbai",
            trip_destination="Goa",
            trip_end_date=date(2026, 12, 24),
            traveler_count=2,
        )
        result = make_service(output).extract("Mumbai to Goa ending Dec 24")
        fields = {cl.field for cl in result.clarification_needed}
        assert "travel.timing" in fields
        assert result.requirements.trip.dates is None

    def test_start_and_end_is_valid_date_range(self, make_service):
        output = LLMExtractionOutput(
            trip_origin="Mumbai",
            trip_destination="Goa",
            trip_start_date=date(2026, 12, 20),
            trip_end_date=date(2026, 12, 24),
            traveler_count=2,
        )
        result = make_service(output).extract("Mumbai to Goa Dec 20-24")
        fields = {cl.field for cl in result.clarification_needed}
        assert "travel.timing" not in fields
        assert result.requirements.trip.dates.date_range.start == date(2026, 12, 20)
        assert result.requirements.trip.dates.date_range.end == date(2026, 12, 24)

    def test_duration_only_is_valid_timing(self, make_service):
        output = LLMExtractionOutput(
            trip_origin="Mumbai",
            trip_destination="Goa",
            duration_days=3,
            traveler_count=2,
        )
        result = make_service(output).extract("Mumbai to Goa for 3 days")
        fields = {cl.field for cl in result.clarification_needed}
        assert "travel.timing" not in fields
        assert result.requirements.trip.dates.duration_days == 3