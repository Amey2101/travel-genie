from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from app.models.travel_requirements import (
    TimePreference,
    TransportMode,
    TravelRequirements,
)


class ResultSource(str, Enum):
    """Provenance of a provider result.

    The safe default is ``DEMO``: a result is only treated as *verified* when a
    provider explicitly says so. Synthetic or estimated data must never be
    presented as live/verified inventory.
    """

    VERIFIED = "verified"
    ESTIMATE = "estimate"
    DEMO = "demo"


class SearchError(Exception):
    """A bookable-inventory provider could not return usable results.

    The message is safe to surface to clients: it never contains credentials or
    raw provider payloads. Kept distinct from a successful empty search.
    """

    def __init__(self, message: str, *, code: str = "search_unavailable") -> None:
        super().__init__(message)
        self.code = code


class SearchRequest(BaseModel):
    """Handoff contract for the travel search capability."""

    workflow_id: str
    requirements: TravelRequirements


class SearchResult(BaseModel):
    """One candidate found by a provider.

    Fields are present only when real data is available — never fabricated.
    ``source`` records provenance so downstream code (and clients) can tell
    verified provider data from estimates or clearly-labelled demo fixtures.
    ``bookable`` is ``False`` for any result that is not a confirmed,
    purchasable product (e.g. synthetic demo data).
    """

    provider_ref: str
    label: str
    price: Decimal | None = None
    currency: str | None = None
    source: ResultSource = ResultSource.DEMO
    bookable: bool = False
    category: str | None = None
    transport_mode: TransportMode | None = None
    duration_days: int | None = None
    departure_time: TimePreference | None = None
    arrival_time: TimePreference | None = None
    start_date: date | None = None
    end_date: date | None = None


class SearchResponse(BaseModel):
    workflow_id: str
    results: list[SearchResult] = Field(default_factory=list)


@runtime_checkable
class SearchProvider(Protocol):
    """External travel search boundary for bookable inventory.

    Kept distinct from ``DiscoveryProvider`` (destination places/attractions).
    Implementations must raise ``SearchError`` for provider failures rather
    than returning an empty response.
    """

    def search(self, request: SearchRequest) -> SearchResponse: ...
