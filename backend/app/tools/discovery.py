from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class DiscoveryError(Exception):
    """Destination discovery could not return usable results.

    The message is safe to surface to clients: it never contains credentials or
    raw provider payloads.
    """


class DiscoveryResult(BaseModel):
    """One place/attraction returned by a discovery provider.

    This is *destination* content (things to see and do). It is deliberately
    separate from ``app.tools.search.SearchResult``, which models bookable
    transport/accommodation inventory. A discovery result is not a bookable
    travel product and carries no availability or bookable price.
    """

    title: str
    category: str = "Place"
    address: str = ""
    rating: float | None = None
    reviews: int | None = None
    price: str = ""
    description: str = ""
    hours: str = ""
    thumbnail: str = ""
    website: str = ""
    maps_url: str = ""


class DiscoveryResponse(BaseModel):
    destination: str
    category: str
    query: str
    count: int
    results: list[DiscoveryResult] = Field(default_factory=list)


@runtime_checkable
class DiscoveryProvider(Protocol):
    """External destination-discovery boundary.

    Kept distinct from ``SearchProvider`` (bookable inventory) on purpose.
    """

    async def search_places(
        self, destination: str, category: str
    ) -> DiscoveryResponse: ...
