"""Deterministic demo ``SearchProvider``.

This provider exists so the planning pipeline can be exercised locally and in
tests without a live bookable-inventory integration. Its output is **synthetic
fixture data**:

- every result is labelled ``source=ResultSource.DEMO`` and ``bookable=False``;
- it never claims prices, schedules or availability are live or real;
- it never reads the user's budget to make a result appear to fit;
- prices are a deterministic function of the route only, so repeated equivalent
  requests are stable.

It is a development default only, selected explicitly via ``SEARCH_PROVIDER``.
It is deliberately *not* a ``DiscoveryProvider`` and does not reuse Google Maps
attractions as transport/accommodation inventory.
"""

from __future__ import annotations

import hashlib
from datetime import date
from decimal import Decimal

from app.models.travel_requirements import TimePreference, TransportMode
from app.tools.search import (
    ResultSource,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

# Synthetic availability window. Deliberately a fixed constant (independent of
# the request and of the user's dates) so it is obvious this is not real
# inventory and never "fitted" to the request.
DEMO_AVAILABILITY_START = date(2026, 1, 1)
DEMO_AVAILABILITY_END = date(2027, 12, 31)

# (mode, display name, price multiplier, trip duration in days, departure, arrival)
_DEMO_MODES: tuple[
    tuple[TransportMode, str, float, int, TimePreference, TimePreference], ...
] = (
    (TransportMode.FLIGHT, "Flight", 3.0, 3, TimePreference.MORNING, TimePreference.AFTERNOON),
    (TransportMode.TRAIN, "Train", 1.4, 5, TimePreference.EARLY_MORNING, TimePreference.EVENING),
    (TransportMode.BUS, "Bus", 0.8, 6, TimePreference.EVENING, TimePreference.MORNING),
    (TransportMode.CAR, "Car", 1.1, 4, TimePreference.MORNING, TimePreference.AFTERNOON),
)

_BASE_PRICE = 4000
_BASE_PRICE_SPAN = 6000
DEFAULT_CURRENCY = "INR"


def _seed(text: str) -> int:
    """Stable, process-independent integer derived from ``text``.

    Uses SHA-256 rather than ``hash()`` because the built-in hash is salted per
    process and would break determinism across runs.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


class DemoSearchProvider:
    """Synchronous, deterministic, clearly-labelled demo inventory provider."""

    def search(self, request: SearchRequest) -> SearchResponse:
        requirements = request.requirements
        origin = (requirements.trip.origin or "").strip()
        destination = (requirements.trip.destination or "").strip()
        currency = requirements.budget.currency or DEFAULT_CURRENCY

        route = f"{origin.lower()}|{destination.lower()}"
        base_price = _BASE_PRICE + (_seed(f"price:{route}") % _BASE_PRICE_SPAN)

        results: list[SearchResult] = []
        for mode, name, multiplier, duration, depart, arrive in _DEMO_MODES:
            ref_seed = _seed(f"ref:{route}:{mode.value}") % 100000
            price = int(round(base_price * multiplier))
            results.append(
                SearchResult(
                    provider_ref=f"demo-{mode.value}-{ref_seed:05d}",
                    label=f"Demo {name}: {origin or 'origin'} to "
                    f"{destination or 'destination'}",
                    price=Decimal(price),
                    currency=currency,
                    source=ResultSource.DEMO,
                    bookable=False,
                    category="transport",
                    transport_mode=mode,
                    duration_days=duration,
                    departure_time=depart,
                    arrival_time=arrive,
                    start_date=DEMO_AVAILABILITY_START,
                    end_date=DEMO_AVAILABILITY_END,
                )
            )
        return SearchResponse(workflow_id=request.workflow_id, results=results)
