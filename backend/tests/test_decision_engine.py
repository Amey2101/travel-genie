"""Tests for the deterministic ``DecisionEngine`` implementation."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.decision.deterministic import DeterministicDecisionEngine, RejectionReason
from app.decision.engine import (
    Candidate,
    RecommendationInput,
    RecommendationStatus,
)
from app.models.travel_requirements import (
    BudgetDetails,
    ContextualSignals,
    DateRange,
    HardConstraints,
    SoftPreferences,
    TimePreference,
    TransportMode,
    TravelerDetails,
    TravelRequirements,
    TripDetails,
)
from app.tools.search import ResultSource


def _requirements(
    *,
    budget_total: Decimal | None = None,
    hard: HardConstraints | None = None,
    soft: SoftPreferences | None = None,
) -> TravelRequirements:
    return TravelRequirements(
        trip=TripDetails(origin="Mumbai", destination="Goa"),
        travelers=TravelerDetails(count=2),
        budget=BudgetDetails(total=budget_total, currency="INR"),
        hard_constraints=hard or HardConstraints(),
        soft_preferences=soft or SoftPreferences(),
        contextual_signals=ContextualSignals(),
    )


def _candidate(ref: str = "c1", **kwargs) -> Candidate:
    return Candidate(provider_ref=ref, label=f"Option {ref}", **kwargs)


def _recommend(requirements: TravelRequirements, candidates: list[Candidate]):
    return DeterministicDecisionEngine().recommend(
        RecommendationInput(
            workflow_id="wf-1",
            requirements=requirements,
            candidates=candidates,
        )
    )


class TestOutcomes:
    def test_empty_candidates_is_no_results(self) -> None:
        rec = _recommend(_requirements(), [])
        assert rec.status is RecommendationStatus.NO_RESULTS
        assert rec.options == []

    def test_all_rejected_is_no_feasible_options(self) -> None:
        req = _requirements(hard=HardConstraints(max_budget=Decimal("100")))
        rec = _recommend(req, [_candidate("a", price=Decimal("500"))])
        assert rec.status is RecommendationStatus.NO_FEASIBLE_OPTIONS
        assert rec.options == []
        assert rec.rejected
        assert (
            RejectionReason.MAX_BUDGET_EXCEEDED.value
            in rec.rejected[0].reasons
        )

    def test_feasible_candidates_yield_ok_ranking(self) -> None:
        req = _requirements(budget_total=Decimal("10000"))
        rec = _recommend(
            req,
            [
                _candidate("a", price=Decimal("9000")),
                _candidate("b", price=Decimal("4000")),
            ],
        )
        assert rec.status is RecommendationStatus.OK
        assert [o.candidate.provider_ref for o in rec.options] == ["b", "a"]
        assert [o.rank for o in rec.options] == [1, 2]


class TestHardConstraints:
    def test_unknown_price_under_hard_budget_is_rejected(self) -> None:
        req = _requirements(hard=HardConstraints(max_budget=Decimal("10000")))
        rec = _recommend(req, [_candidate("a", price=None)])
        assert rec.status is RecommendationStatus.NO_FEASIBLE_OPTIONS
        assert (
            RejectionReason.MAX_BUDGET_UNVERIFIED.value
            in rec.rejected[0].reasons
        )

    def test_price_within_hard_budget_passes(self) -> None:
        req = _requirements(hard=HardConstraints(max_budget=Decimal("10000")))
        rec = _recommend(req, [_candidate("a", price=Decimal("10000"))])
        assert rec.status is RecommendationStatus.OK

    def test_required_transport_mismatch(self) -> None:
        req = _requirements(
            hard=HardConstraints(required_transport=TransportMode.FLIGHT)
        )
        rec = _recommend(
            req,
            [
                _candidate("a", transport_mode=TransportMode.TRAIN),
                _candidate("b", transport_mode=TransportMode.FLIGHT),
            ],
        )
        assert [o.candidate.provider_ref for o in rec.options] == ["b"]
        assert RejectionReason.TRANSPORT_MISMATCH.value in rec.rejected[0].reasons

    def test_required_transport_unverified_is_rejected(self) -> None:
        req = _requirements(
            hard=HardConstraints(required_transport=TransportMode.FLIGHT)
        )
        rec = _recommend(req, [_candidate("a", transport_mode=None)])
        assert RejectionReason.TRANSPORT_UNVERIFIED.value in rec.rejected[0].reasons

    def test_min_and_max_duration(self) -> None:
        req = _requirements(
            hard=HardConstraints(min_duration_days=3, max_duration_days=5)
        )
        rec = _recommend(
            req,
            [
                _candidate("short", duration_days=2),
                _candidate("ok", duration_days=4),
                _candidate("long", duration_days=6),
            ],
        )
        assert [o.candidate.provider_ref for o in rec.options] == ["ok"]
        reasons = {rc.candidate.provider_ref: rc.reasons for rc in rec.rejected}
        assert RejectionReason.DURATION_BELOW_MIN.value in reasons["short"]
        assert RejectionReason.DURATION_ABOVE_MAX.value in reasons["long"]

    def test_duration_unverified_reported_once(self) -> None:
        req = _requirements(
            hard=HardConstraints(min_duration_days=3, max_duration_days=5)
        )
        rec = _recommend(req, [_candidate("a", duration_days=None)])
        assert rec.rejected[0].reasons.count(
            RejectionReason.DURATION_UNVERIFIED.value
        ) == 1

    def test_departure_time_constraint(self) -> None:
        req = _requirements(
            hard=HardConstraints(departure_time=TimePreference.MORNING)
        )
        rec = _recommend(
            req,
            [
                _candidate("am", departure_time=TimePreference.MORNING),
                _candidate("pm", departure_time=TimePreference.EVENING),
            ],
        )
        assert [o.candidate.provider_ref for o in rec.options] == ["am"]
        assert (
            RejectionReason.DEPARTURE_TIME_MISMATCH.value
            in rec.rejected[0].reasons
        )

    def test_arrival_time_constraint(self) -> None:
        req = _requirements(
            hard=HardConstraints(arrival_time=TimePreference.EVENING)
        )
        rec = _recommend(
            req,
            [
                _candidate("am", arrival_time=TimePreference.MORNING),
                _candidate("pm", arrival_time=TimePreference.EVENING),
            ],
        )
        assert [o.candidate.provider_ref for o in rec.options] == ["pm"]
        assert (
            RejectionReason.ARRIVAL_TIME_MISMATCH.value in rec.rejected[0].reasons
        )

    def test_no_preference_time_is_not_enforced(self) -> None:
        req = _requirements(
            hard=HardConstraints(departure_time=TimePreference.NO_PREFERENCE)
        )
        rec = _recommend(req, [_candidate("a", departure_time=None)])
        assert rec.status is RecommendationStatus.OK

    def test_date_window_overlap(self) -> None:
        req = _requirements(
            hard=HardConstraints(
                date_range=DateRange(start=date(2026, 6, 1), end=date(2026, 6, 30))
            )
        )
        rec = _recommend(
            req,
            [
                _candidate(
                    "overlap",
                    start_date=date(2026, 5, 20),
                    end_date=date(2026, 6, 10),
                ),
                _candidate(
                    "outside",
                    start_date=date(2026, 8, 1),
                    end_date=date(2026, 9, 1),
                ),
            ],
        )
        assert [o.candidate.provider_ref for o in rec.options] == ["overlap"]
        assert RejectionReason.DATE_UNAVAILABLE.value in rec.rejected[0].reasons

    def test_date_window_unverified(self) -> None:
        req = _requirements(
            hard=HardConstraints(
                date_range=DateRange(start=date(2026, 6, 1), end=date(2026, 6, 30))
            )
        )
        rec = _recommend(req, [_candidate("a", start_date=None, end_date=None)])
        assert RejectionReason.DATE_UNVERIFIED.value in rec.rejected[0].reasons


class TestRankingAndExplanation:
    def test_deterministic_order_and_stable_tie_break(self) -> None:
        req = _requirements()
        rec = _recommend(
            req,
            [_candidate("zeta"), _candidate("alpha"), _candidate("beta")],
        )
        assert [o.candidate.provider_ref for o in rec.options] == [
            "alpha",
            "beta",
            "zeta",
        ]

    def test_tie_break_is_repeatable(self) -> None:
        req = _requirements()
        candidates = [_candidate("b"), _candidate("a")]
        first = _recommend(req, candidates)
        second = _recommend(req, candidates)
        assert first.model_dump() == second.model_dump()

    def test_soft_preference_changes_ranking(self) -> None:
        req = _requirements(
            soft=SoftPreferences(preferred_transport=TransportMode.FLIGHT)
        )
        rec = _recommend(
            req,
            [
                _candidate("train", transport_mode=TransportMode.TRAIN),
                _candidate("flight", transport_mode=TransportMode.FLIGHT),
            ],
        )
        assert rec.options[0].candidate.provider_ref == "flight"
        assert rec.options[0].score > rec.options[1].score

    def test_explanations_reference_actual_values(self) -> None:
        req = _requirements(
            budget_total=Decimal("10000"),
            soft=SoftPreferences(preferred_transport=TransportMode.FLIGHT),
        )
        rec = _recommend(
            req,
            [
                _candidate(
                    "flight",
                    price=Decimal("9000"),
                    currency="INR",
                    transport_mode=TransportMode.FLIGHT,
                    source=ResultSource.DEMO,
                )
            ],
        )
        reasons = " ".join(rec.options[0].reasons)
        assert "9000" in reasons
        assert "flight" in reasons
        assert "demo" in reasons.lower()

    def test_unknown_price_is_not_scored_but_not_imputed(self) -> None:
        req = _requirements(budget_total=Decimal("10000"))
        rec = _recommend(req, [_candidate("a", price=None)])
        assert rec.status is RecommendationStatus.OK
        assert "price unavailable" in " ".join(rec.options[0].reasons)


class TestSafetyBoundaries:
    def test_engine_does_not_depend_on_an_llm(self) -> None:
        import app.decision.deterministic as deterministic

        assert not hasattr(deterministic, "LLMClient")

    def test_recommendation_carries_no_approval_state(self) -> None:
        rec = _recommend(
            _requirements(), [_candidate("a", price=Decimal("100"))]
        )
        assert "approval" not in rec.model_dump()
        assert "approval" not in rec.model_dump_json()

    def test_engine_does_not_reference_approval_or_execution(self) -> None:
        import app.decision.deterministic as deterministic

        assert not hasattr(deterministic, "Approval")
        assert not hasattr(deterministic, "ExecutionProvider")
