"""Deterministic budget estimation.

This module adapts the useful, purely-deterministic cost helpers from the
teammate's original ``services/planner.py`` (``heuristic_lines`` and
``compute_budget``) into our architecture. It is intentionally **not** the
teammate's planner: no ``build_plan``/``fallback_plan``, no itinerary clients,
no ``PlanResponse``, and no LLM or network calls.

Responsibilities:
1. Turn already-extracted ``TravelRequirements`` (plus, when justified, a
   recommended candidate's verified price) into a typed estimator input.
2. Produce a transparent, per-category estimate with an explicit total,
   target comparison, and machine-readable provenance.

Invariants:
- The estimate is deterministic: identical input always yields identical output.
- Every number is an *estimate*, never a verified provider price or a booking.
- Unknown metrics are never imputed and no price is invented. A synthetic demo
  price is never represented as a real fare.
- Budget estimation is deliberately separate from recommendation ranking and
  never influences filtering, scoring, approval, or execution.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator

from app.decision.engine import Recommendation, RecommendationStatus
from app.models.travel_requirements import (
    ComfortLevel,
    TravelRequirements,
    TripDates,
)
from app.tools.search import ResultSource

# ---------------------------------------------------------------------------
# Estimator input / output contract
# ---------------------------------------------------------------------------


class StyleTier(str, Enum):
    """Normalised travel style used by the estimator.

    Our ``ComfortLevel`` has five levels; the fixed cost table has three tiers.
    The mapping is explicit in ``style_tier`` (budget/economy -> budget,
    standard -> balanced, premium/luxury -> luxury).
    """

    BUDGET = "budget"
    BALANCED = "balanced"
    LUXURY = "luxury"


class TransportationSource(str, Enum):
    """Where a ``transportation`` line came from.

    ``HEURISTIC`` is a rule-of-thumb assumption; ``PROVIDER_VERIFIED`` means the
    line reuses a selected candidate's price whose provenance is ``verified``.
    Demo/estimate prices are never used as a real fare.
    """

    HEURISTIC = "heuristic"
    PROVIDER_VERIFIED = "provider_verified"


def _normalise_currency(value: str | None) -> str:
    """Trim a currency code and convert it to upper case.

    Mirrors how currency codes are validated elsewhere (``SUPPORTED_CURRENCIES``
    in ``app/services/structured_requirements.py``), so ``"usd"`` and ``"USD"``
    mean the same thing throughout the estimate.
    """
    if value is None:
        return ""
    return str(value).strip().upper()


class BudgetEstimateInput(BaseModel):
    """Typed, already-resolved input to a ``BudgetEstimator``.

    The mapping from our domain lives in ``budget_input_from_requirements``; it
    keeps estimator inputs free of domain ambiguity (see the date/duration note
    there).
    """

    origin: str = ""
    destination: str = ""
    currency: str = "INR"
    travelers: int = Field(default=1, ge=1)
    days: int = Field(ge=1)
    nights: int = Field(ge=0)
    style: StyleTier = StyleTier.BALANCED
    target_budget: Decimal | None = None
    target_is_approximate: bool = False
    hard_max_budget: Decimal | None = None
    transportation_price: Decimal | None = None
    transportation_source: ResultSource | None = None

    @field_validator("currency")
    @classmethod
    def _normalise_currency_code(cls, value: str) -> str:
        return _normalise_currency(value)


class BudgetLine(BaseModel):
    """One estimated cost category. ``amount`` is in ``BudgetEstimate.currency``."""

    key: str
    label: str
    amount: Decimal


class BudgetEstimate(BaseModel):
    """Transparent, per-category budget estimate for a trip.

    ``difference`` is ``target_budget - estimated_total`` (``None`` when no
    target was provided). ``within_target`` compares against the *target*, which
    is a soft reference; ``hard_max_budget`` is surfaced separately and is never
    conflated with the target.
    """

    currency: str
    lines: list[BudgetLine] = Field(default_factory=list)
    estimated_total: Decimal
    target_budget: Decimal | None = None
    target_is_approximate: bool = False
    hard_max_budget: Decimal | None = None
    difference: Decimal | None = None
    within_target: bool | None = None
    hard_max_exceeded: bool | None = None
    per_person: Decimal
    transportation_source: TransportationSource = TransportationSource.HEURISTIC
    suggestions: list[str] = Field(default_factory=list)
    note: str
    disclaimer: str


@runtime_checkable
class BudgetEstimator(Protocol):
    """Deterministic budget-estimation boundary.

    Contract: no LLM, no network, deterministic and transparent. The estimate is
    informational only and must never be treated as a verified price, an
    approval, or a booking.
    """

    def estimate(self, input_: BudgetEstimateInput) -> BudgetEstimate: ...


# ---------------------------------------------------------------------------
# Fixed assumptions (adapted from the teammate's services/planner.py)
#
# These are rough planning assumptions expressed in INR, NOT live prices or
# exchange rates. They are deliberately constants so results are reproducible.
# ---------------------------------------------------------------------------

CATEGORY_LABELS: tuple[tuple[str, str], ...] = (
    ("transportation", "Transportation"),
    ("accommodation", "Accommodation"),
    ("food", "Food"),
    ("activities", "Activities and sightseeing"),
    ("miscellaneous", "Miscellaneous"),
)

_RATES_FROM_INR: dict[str, Decimal] = {
    "INR": Decimal("1.0"),
    "USD": Decimal("0.012"),
    "EUR": Decimal("0.011"),
    "GBP": Decimal("0.0095"),
    "AED": Decimal("0.044"),
    "SGD": Decimal("0.016"),
    "THB": Decimal("0.40"),
}

_STYLE_COSTS: dict[StyleTier, dict[str, int]] = {
    StyleTier.BUDGET: {
        "intercity": 2000,
        "local_day": 400,
        "room": 1200,
        "food": 500,
        "act": 300,
    },
    StyleTier.BALANCED: {
        "intercity": 3500,
        "local_day": 700,
        "room": 2500,
        "food": 800,
        "act": 600,
    },
    StyleTier.LUXURY: {
        "intercity": 8000,
        "local_day": 2000,
        "room": 7000,
        "food": 2500,
        "act": 2000,
    },
}

_MISC_RATE = Decimal("0.08")
_LOWER_STYLE = {StyleTier.LUXURY: StyleTier.BALANCED, StyleTier.BALANCED: StyleTier.BUDGET}

NOTE = (
    "Estimates only. Based on typical cost ranges for your travel style, "
    "not live prices."
)
DISCLAIMER = (
    "These figures are planning estimates, not verified provider prices or "
    "bookings. Nothing has been reserved or purchased."
)


def _rate_from_inr(currency: str) -> Decimal:
    """Rate for a *supported* currency.

    An unsupported code must never fall back to ``1.``: that would silently
    present an INR-based figure as if it were denominated in ``currency`` (and
    would let unsupported codes inherit INR rounding). Instead of a misleading
    estimate, the unsupported code is reported as an error, which callers
    (``budget_input_from_requirements``, ``TravelAgent._run_budget_estimate``)
    translate into *no* estimate.
    """
    try:
        return _RATES_FROM_INR[currency]
    except KeyError:
        raise ValueError(
            f"Unsupported currency for budget estimation: {currency!r}. "
            f"Supported: {', '.join(sorted(_RATES_FROM_INR))}."
        ) from None


def _step(currency: str) -> Decimal:
    return Decimal(10) if _rate_from_inr(currency) >= Decimal("0.3") else Decimal(1)


def _round_money(value: Decimal, currency: str) -> Decimal:
    """Round to a currency-appropriate step, mirroring the teammate's helper."""
    step = _step(currency)
    rounded = (value / step).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN)
    return rounded * step


def _money(amount: Decimal, currency: str) -> str:
    return f"{currency} {amount.quantize(Decimal('1'), rounding=ROUND_HALF_UP):,}"


def style_tier(comfort: ComfortLevel | None) -> StyleTier:
    """Map our five-level ``ComfortLevel`` onto the estimator's three tiers."""
    if comfort in (ComfortLevel.BUDGET, ComfortLevel.ECONOMY):
        return StyleTier.BUDGET
    if comfort in (ComfortLevel.PREMIUM, ComfortLevel.LUXURY):
        return StyleTier.LUXURY
    return StyleTier.BALANCED


class StyleBasedBudgetEstimator:
    """Deterministic, style-based budget estimator.

    ``heuristic_lines`` is adapted from the teammate's planner; ``compute_budget``
    assembled the totals/status/suggestions. Both are re-implemented here against
    typed models, without importing the teammate's module.
    """

    def estimate(self, input_: BudgetEstimateInput) -> BudgetEstimate:
        lines = self._heuristic_lines(input_)

        transportation_source = TransportationSource.HEURISTIC
        if (
            input_.transportation_price is not None
            and input_.transportation_source is ResultSource.VERIFIED
        ):
            # Only a *verified* provider price may be reused as a real
            # transportation input; demo/estimate prices are never treated as
            # live fares. Miscellaneous is recomputed so the total stays the
            # exact sum of the category lines.
            lines["transportation"] = _round_money(
                input_.transportation_price, input_.currency
            )
            lines["miscellaneous"] = _round_money(
                _MISC_RATE * sum(lines.values()), input_.currency
            )
            transportation_source = TransportationSource.PROVIDER_VERIFIED

        total = sum(lines.values(), Decimal(0))

        difference: Decimal | None = None
        within_target: bool | None = None
        if input_.target_budget is not None:
            difference = input_.target_budget - total
            within_target = difference >= 0

        hard_max_exceeded: bool | None = None
        if input_.hard_max_budget is not None:
            hard_max_exceeded = total > input_.hard_max_budget

        per_person = _round_money(
            total / Decimal(input_.travelers), input_.currency
        )

        return BudgetEstimate(
            currency=input_.currency,
            lines=[
                BudgetLine(key=key, label=label, amount=lines[key])
                for key, label in CATEGORY_LABELS
            ],
            estimated_total=total,
            target_budget=input_.target_budget,
            target_is_approximate=input_.target_is_approximate,
            hard_max_budget=input_.hard_max_budget,
            difference=difference,
            within_target=within_target,
            hard_max_exceeded=hard_max_exceeded,
            per_person=per_person,
            transportation_source=transportation_source,
            suggestions=self._suggestions(input_, lines, total),
            note=NOTE,
            disclaimer=DISCLAIMER,
        )

    # -- adapted from the teammate's heuristic_lines ------------------------
    def _heuristic_lines(self, input_: BudgetEstimateInput) -> dict[str, Decimal]:
        costs = _STYLE_COSTS[input_.style]
        fx = _rate_from_inr(input_.currency)
        days = input_.days
        nights = input_.nights
        travelers = input_.travelers
        same_city = input_.origin.strip().lower() == input_.destination.strip().lower()
        rooms = Decimal(-(-travelers // 2))  # ceil(travelers / 2)
        raw: dict[str, Decimal] = {
            "transportation": Decimal(
                (0 if same_city else costs["intercity"] * travelers)
                + costs["local_day"] * days
            ),
            "accommodation": Decimal(costs["room"]) * rooms * nights,
            "food": Decimal(costs["food"]) * travelers * days,
            "activities": Decimal(costs["act"]) * travelers * days,
        }
        raw["miscellaneous"] = _MISC_RATE * sum(raw.values(), Decimal(0))
        return {key: _round_money(value * fx, input_.currency) for key, value in raw.items()}

    # -- adapted from the teammate's compute_budget -------------------------
    def _suggestions(
        self,
        input_: BudgetEstimateInput,
        lines: dict[str, Decimal],
        total: Decimal,
    ) -> list[str]:
        currency = input_.currency
        suggestions: list[str] = []

        if input_.hard_max_budget is not None and total > input_.hard_max_budget:
            suggestions.append(
                f"This estimate is above your stated maximum budget of "
                f"{_money(input_.hard_max_budget, currency)}."
            )

        if input_.target_budget is None:
            suggestions.append(
                "No target budget was provided, so this estimate is shown for "
                "reference only."
            )
            return suggestions

        difference = input_.target_budget - total
        if difference >= 0:
            suggestions.append(
                f"Estimated spend leaves about {_money(difference, currency)} of "
                "your target as a buffer. Keep some for surprises."
            )
            return suggestions

        suggestions.append(
            f"The estimate is about {_money(-difference, currency)} over your target."
        )
        lower = _LOWER_STYLE.get(input_.style)
        if lower is not None:
            alt = sum(
                self._heuristic_lines(input_.model_copy(update={"style": lower})).values(),
                Decimal(0),
            )
            suggestions.append(
                f"Switching to a {lower.value} travel style would bring a rough "
                f"estimate to about {_money(alt, currency)}."
            )
        if total > 0 and lines["transportation"] > Decimal("0.25") * total:
            suggestions.append(
                "Compare trains or buses with flights for the journey and book early."
            )
        if input_.days > 1:
            saving = _round_money(total / Decimal(input_.days), currency)
            suggestions.append(
                f"Cutting one day would save roughly {_money(saving, currency)}."
            )
        suggestions.append(
            "Pick guesthouses or hostels, eat where locals eat, and favour free or "
            "low-cost sights."
        )
        return suggestions


# ---------------------------------------------------------------------------
# Domain mapping
# ---------------------------------------------------------------------------


def trip_days_and_nights(dates: TripDates | None) -> tuple[int, int] | None:
    """Resolve trip length from our domain's ``TripDates``.

    Date/duration semantics (explicit and deliberate):

    - A concrete ``date_range`` is counted **inclusively** — the return date is
      itself a travel day — so ``days = (end - start).days + 1`` and
      ``nights = days - 1``. This matches the teammate's convention and how a
      traveller describes "20th to 24th" (5 days, 4 nights).
    - Otherwise a stated ``duration_days`` is the number of days away, with
      ``nights = duration_days - 1``.

    Our requirement-extraction transform defines ``duration_days`` as
    ``(end - start).days`` when both are supplied (an exclusive count). The
    estimator does **not** reinterpret or mutate that stored value: when a
    concrete range exists it derives an inclusive count from the dates, and only
    falls back to ``duration_days`` when no range is available. This keeps
    existing extraction/clarification semantics untouched.
    """
    if not isinstance(dates, TripDates):
        return None
    if dates.date_range is not None:
        span = (dates.date_range.end - dates.date_range.start).days
        if span < 0:
            return None
        days = span + 1
    elif dates.duration_days is not None:
        days = dates.duration_days
    else:
        return None
    if days < 1:
        return None
    return days, max(days - 1, 0)


def _selected_verified_transportation(
    recommendation: Recommendation | None, currency: str
) -> tuple[Decimal, ResultSource] | None:
    """Return the top option's price only when it is a verified transport fare."""
    if recommendation is None or recommendation.status is not RecommendationStatus.OK:
        return None
    if not recommendation.options:
        return None
    candidate = recommendation.options[0].candidate
    if candidate.price is None or candidate.transport_mode is None:
        return None
    if candidate.source is not ResultSource.VERIFIED:
        return None
    if candidate.currency is not None and candidate.currency != currency:
        return None
    return candidate.price, candidate.source


def budget_input_from_requirements(
    requirements: TravelRequirements,
    recommendation: Recommendation | None = None,
) -> BudgetEstimateInput | None:
    """Map our domain + a recommendation into a typed estimator input.

    Returns ``None`` when no meaningful estimate can be calculated — missing
    timing or traveller count, or an explicitly stated currency the estimator
    does not support — rather than inventing assumptions or a 1:1 conversion.
    The caller represents this explicitly by leaving the estimate absent.

    Currency codes are normalised (trimmed, upper-cased) here, matching how
    ``SUPPORTED_CURRENCIES`` validates them elsewhere in the project.
    """
    travelers = requirements.travelers.count
    if travelers is None or travelers < 1:
        return None

    dates = requirements.trip.dates
    length = trip_days_and_nights(dates)
    if length is None:
        return None
    days, nights = length

    budget = requirements.budget
    currency = _normalise_currency(budget.currency) or "INR"
    if currency not in _RATES_FROM_INR:
        # An unsupported code must not silently fall back to a 1:1 INR rate:
        # simply produce no estimate.
        return None
    target = budget.total
    hard_max = requirements.hard_constraints.max_budget

    transportation = _selected_verified_transportation(recommendation, currency)

    return BudgetEstimateInput(
        origin=requirements.trip.origin or "",
        destination=requirements.trip.destination or "",
        currency=currency,
        travelers=travelers,
        days=days,
        nights=nights,
        style=style_tier(requirements.soft_preferences.comfort_level),
        target_budget=target,
        target_is_approximate=budget.is_approximate,
        hard_max_budget=hard_max,
        transportation_price=transportation[0] if transportation else None,
        transportation_source=transportation[1] if transportation else None,
    )
