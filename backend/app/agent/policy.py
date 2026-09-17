from __future__ import annotations

from app.models.travel_requirements import (
    ClarificationImportance,
    ClarificationItem,
    ExtractionResult,
)


def missing_required(result: ExtractionResult) -> list[ClarificationItem]:
    """Clarifications that must be resolved before requirements are actionable."""
    return [
        item
        for item in result.clarification_needed
        if item.importance is ClarificationImportance.REQUIRED
    ]


def requirements_actionable(result: ExtractionResult) -> bool:
    """True when no required clarification remains open."""
    return not missing_required(result)