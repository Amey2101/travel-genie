"""Deterministic decision engine.

Responsibilities, in strict order:

1. Enforce hard constraints (filter) before any scoring.
2. Rank feasible candidates with transparent, deterministic criteria.
3. Explain both recommendations and rejections with machine-readable reasons.

The engine never calls an LLM, never imputes unknown metrics, and never makes
approval or execution decisions.
"""

from __future__ import annotations

from enum import Enum

from app.decision.engine import (
    Candidate,
    Recommendation,
    RecommendationInput,
    RecommendationStatus,
    RecommendedOption,
    RejectedCandidate,
)
from app.models.travel_requirements import (
    TimePreference,
    TransportMode,
    TravelRequirements,
)


class RejectionReason(str, Enum):
    """Machine-readable reasons a candidate failed a hard constraint."""

    MAX_BUDGET_EXCEEDED = "max_budget_exceeded"
    MAX_BUDGET_UNVERIFIED = "max_budget_unverified"
    TRANSPORT_MISMATCH = "transport_mismatch"
    TRANSPORT_UNVERIFIED = "transport_unverified"
    DURATION_BELOW_MIN = "duration_below_min"
    DURATION_ABOVE_MAX = "duration_above_max"
    DURATION_UNVERIFIED = "duration_unverified"
    DEPARTURE_TIME_MISMATCH = "departure_time_mismatch"
    DEPARTURE_TIME_UNVERIFIED = "departure_time_unverified"
    ARRIVAL_TIME_MISMATCH = "arrival_time_mismatch"
    ARRIVAL_TIME_UNVERIFIED = "arrival_time_unverified"
    DATE_UNAVAILABLE = "date_unavailable"
    DATE_UNVERIFIED = "date_unverified"


# Transparent scoring weights. A component only contributes when the matching
# user preference/target is present; missing candidate metrics score 0 and are
# never imputed.
_SCORE_WEIGHTS = {
    "budget": 0.5,
    "transport": 0.3,
    "departure_time": 0.2,
}


class DeterministicDecisionEngine:
    """Implements the ``DecisionEngine`` contract deterministically."""

    def recommend(self, input_: RecommendationInput) -> Recommendation:
        requirements = input_.requirements
        candidates = input_.candidates

        if not candidates:
            return Recommendation(
                workflow_id=input_.workflow_id,
                status=RecommendationStatus.NO_RESULTS,
            )

        feasible: list[Candidate] = []
        rejected: list[RejectedCandidate] = []
        for candidate in candidates:
            reasons = _constraint_violations(candidate, requirements)
            if reasons:
                rejected.append(RejectedCandidate(candidate=candidate, reasons=reasons))
            else:
                feasible.append(candidate)

        if not feasible:
            return Recommendation(
                workflow_id=input_.workflow_id,
                status=RecommendationStatus.NO_FEASIBLE_OPTIONS,
                rejected=rejected,
            )

        scored = [(_score(candidate, requirements), candidate) for candidate in feasible]
        # Descending score, then stable tie-break by provider_ref for determinism.
        scored.sort(key=lambda pair: (-pair[0], pair[1].provider_ref))

        options = [
            RecommendedOption(
                rank=index + 1,
                candidate=candidate,
                score=score,
                reasons=_explain(candidate, requirements),
            )
            for index, (score, candidate) in enumerate(scored)
        ]
        return Recommendation(
            workflow_id=input_.workflow_id,
            status=RecommendationStatus.OK,
            options=options,
            rejected=rejected,
        )


# ---------------------------------------------------------------------------
# Hard constraints
# ---------------------------------------------------------------------------


def _constraint_violations(
    candidate: Candidate, requirements: TravelRequirements
) -> list[str]:
    """Return machine-readable reasons the candidate violates a hard constraint.

    A constraint that cannot be evaluated against the candidate's data is not
    silently assumed to pass: the candidate is rejected with an explicit
    ``*_unverified`` reason.
    """
    hard = requirements.hard_constraints
    reasons: list[str] = []

    if hard.max_budget is not None:
        if candidate.price is None:
            reasons.append(RejectionReason.MAX_BUDGET_UNVERIFIED.value)
        elif candidate.price > hard.max_budget:
            reasons.append(RejectionReason.MAX_BUDGET_EXCEEDED.value)

    if hard.required_transport is not None and hard.required_transport is not TransportMode.ANY:
        if candidate.transport_mode is None:
            reasons.append(RejectionReason.TRANSPORT_UNVERIFIED.value)
        elif candidate.transport_mode is not hard.required_transport:
            reasons.append(RejectionReason.TRANSPORT_MISMATCH.value)

    if hard.min_duration_days is not None:
        if candidate.duration_days is None:
            reasons.append(RejectionReason.DURATION_UNVERIFIED.value)
        elif candidate.duration_days < hard.min_duration_days:
            reasons.append(RejectionReason.DURATION_BELOW_MIN.value)

    if hard.max_duration_days is not None:
        if candidate.duration_days is None:
            reasons.append(RejectionReason.DURATION_UNVERIFIED.value)
        elif candidate.duration_days > hard.max_duration_days:
            reasons.append(RejectionReason.DURATION_ABOVE_MAX.value)

    if (
        hard.departure_time is not None
        and hard.departure_time is not TimePreference.NO_PREFERENCE
    ):
        if candidate.departure_time is None:
            reasons.append(RejectionReason.DEPARTURE_TIME_UNVERIFIED.value)
        elif candidate.departure_time is not hard.departure_time:
            reasons.append(RejectionReason.DEPARTURE_TIME_MISMATCH.value)

    if (
        hard.arrival_time is not None
        and hard.arrival_time is not TimePreference.NO_PREFERENCE
    ):
        if candidate.arrival_time is None:
            reasons.append(RejectionReason.ARRIVAL_TIME_UNVERIFIED.value)
        elif candidate.arrival_time is not hard.arrival_time:
            reasons.append(RejectionReason.ARRIVAL_TIME_MISMATCH.value)

    if hard.date_range is not None:
        if candidate.start_date is None or candidate.end_date is None:
            reasons.append(RejectionReason.DATE_UNVERIFIED.value)
        elif (
            candidate.end_date < hard.date_range.start
            or candidate.start_date > hard.date_range.end
        ):
            reasons.append(RejectionReason.DATE_UNAVAILABLE.value)

    # Preserve order, drop duplicates (e.g. two duration bounds both unverified).
    return list(dict.fromkeys(reasons))


# ---------------------------------------------------------------------------
# Scoring and explanation
# ---------------------------------------------------------------------------


def _score(candidate: Candidate, requirements: TravelRequirements) -> float:
    """Deterministic score in [0, 1] over the applicable soft criteria."""
    components: dict[str, float] = {}

    budget = requirements.budget.total
    if budget is not None:
        if candidate.price is None or budget <= 0:
            components["budget"] = 0.0
        elif candidate.price <= budget:
            # Within budget: 1.0 at price 0, easing down to 0.5 at the budget.
            components["budget"] = 0.5 + 0.5 * (
                1.0 - float(candidate.price / budget)
            )
        else:
            # Above budget: 0.5 just over budget, easing to 0.0 at 2x budget.
            over = float((candidate.price - budget) / budget)
            components["budget"] = 0.5 * max(0.0, 1.0 - over)

    preferred_transport = requirements.soft_preferences.preferred_transport
    if preferred_transport is not None:
        components["transport"] = (
            1.0
            if candidate.transport_mode is not None
            and candidate.transport_mode is preferred_transport
            else 0.0
        )

    preferred_departure = requirements.soft_preferences.preferred_departure_time
    if preferred_departure is not None:
        components["departure_time"] = (
            1.0
            if candidate.departure_time is not None
            and candidate.departure_time is preferred_departure
            else 0.0
        )

    if not components:
        return 0.0
    total_weight = sum(_SCORE_WEIGHTS[name] for name in components)
    weighted = sum(_SCORE_WEIGHTS[name] * value for name, value in components.items())
    return weighted / total_weight


def _explain(candidate: Candidate, requirements: TravelRequirements) -> list[str]:
    """Concise reasons tied to actual candidate fields and the scoring rules."""
    reasons: list[str] = []

    budget = requirements.budget.total
    if budget is not None:
        if candidate.price is None:
            reasons.append("price unavailable; budget fit not scored")
        elif candidate.price <= budget:
            reasons.append(
                f"price {_money(candidate)} is within the target budget {budget}"
            )
        else:
            reasons.append(
                f"price {_money(candidate)} is above the target budget {budget}"
            )

    preferred_transport = requirements.soft_preferences.preferred_transport
    if preferred_transport is not None:
        if candidate.transport_mode is preferred_transport:
            reasons.append(f"transport matches your preferred {preferred_transport.value}")
        elif candidate.transport_mode is None:
            reasons.append("transport mode unavailable; preference not scored")
        else:
            reasons.append(
                f"transport {candidate.transport_mode.value} does not match your "
                f"preferred {preferred_transport.value}"
            )

    preferred_departure = requirements.soft_preferences.preferred_departure_time
    if preferred_departure is not None:
        if candidate.departure_time is preferred_departure:
            reasons.append(
                f"departs in the {preferred_departure.value.replace('_', ' ')} as preferred"
            )
        elif candidate.departure_time is None:
            reasons.append("departure time unavailable; preference not scored")
        else:
            reasons.append(
                f"departs in the {candidate.departure_time.value.replace('_', ' ')}; "
                f"you preferred {preferred_departure.value.replace('_', ' ')}"
            )

    if candidate.source.value != "verified":
        reasons.append(
            f"source: {candidate.source.value} (synthetic, not bookable)"
        )
    return reasons


def _money(candidate: Candidate) -> str:
    amount = candidate.price
    currency = candidate.currency or ""
    return f"{amount} {currency}".strip()
