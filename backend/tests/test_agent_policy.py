from __future__ import annotations

from app.agent.policy import missing_required, requirements_actionable
from app.models.travel_requirements import (
    BudgetDetails,
    ClarificationImportance,
    ClarificationItem,
    ContextualSignals,
    ExtractionMetadata,
    ExtractionResult,
    HardConstraints,
    SoftPreferences,
    TravelerDetails,
    TravelRequirements,
    TripDetails,
)


def _result(items: list[ClarificationItem]) -> ExtractionResult:
    return ExtractionResult(
        requirements=TravelRequirements(
            trip=TripDetails(),
            travelers=TravelerDetails(),
            budget=BudgetDetails(),
            hard_constraints=HardConstraints(),
            soft_preferences=SoftPreferences(),
            contextual_signals=ContextualSignals(),
        ),
        clarification_needed=items,
        metadata=ExtractionMetadata(original_user_message="x"),
    )


def _item(
    field: str, importance: ClarificationImportance
) -> ClarificationItem:
    return ClarificationItem(
        field=field, importance=importance, question=f"question about {field}"
    )


class TestMissingRequired:
    def test_returns_only_required_items(self) -> None:
        required = _item("trip.origin", ClarificationImportance.REQUIRED)
        recommended = _item("budget.total", ClarificationImportance.RECOMMENDED)
        optional = _item("contextual_signals", ClarificationImportance.OPTIONAL)
        result = _result([required, recommended, optional])
        assert missing_required(result) == [required]

    def test_actionable_when_no_required_items(self) -> None:
        recommended = _item("budget.total", ClarificationImportance.RECOMMENDED)
        assert requirements_actionable(_result([recommended])) is True

    def test_not_actionable_when_required_item_open(self) -> None:
        required = _item("travel.timing", ClarificationImportance.REQUIRED)
        assert requirements_actionable(_result([required])) is False

    def test_actionable_when_no_clarifications(self) -> None:
        assert requirements_actionable(_result([])) is True