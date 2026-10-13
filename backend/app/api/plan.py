"""Compatibility surface for the teammate frontend's one-shot planner.

Milestone 1 only establishes the route so the static frontend's contract is
preserved. The planning pipeline (research + itinerary generation + budget) is
intentionally not implemented yet; until it lands this endpoint answers with an
explicit ``not_implemented`` error rather than fabricating a plan.
"""

from fastapi import APIRouter

from app.api.errors import api_error

router = APIRouter(tags=["planning"])


@router.post("/api/plan")
def plan_compat():
    return api_error(
        501,
        "not_implemented",
        "Trip planning is not available in this build yet. "
        "Conversational requirement collection is available via /workflows.",
    )
