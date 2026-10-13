"""Destination discovery API.

Exposes ``GET /api/discover`` for the Explore page. Uses the ``DiscoveryProvider``
contract (destination places/attractions) — deliberately separate from the
bookable-inventory ``SearchProvider``. No itinerary planning or booking happens
here; the response never claims attractions are bookable travel inventory.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.errors import api_error
from app.providers.serpapi_discovery import SerpApiDiscoveryProvider
from app.tools.discovery import DiscoveryError, DiscoveryProvider, DiscoveryResponse

MAX_DESTINATION = 100
MAX_CATEGORY = 80
DEFAULT_CATEGORY = "tourist attractions"


def discover_router(provider: DiscoveryProvider | None = None) -> APIRouter:
    router = APIRouter(tags=["discovery"])
    active: DiscoveryProvider = provider or SerpApiDiscoveryProvider()

    @router.get("/api/discover", response_model=DiscoveryResponse)
    async def discover(
        destination: str = "", category: str = DEFAULT_CATEGORY
    ) -> DiscoveryResponse | JSONResponse:
        destination = destination.strip()
        category = category.strip()
        if not destination:
            return api_error(
                422, "invalid_destination", "Enter a destination to search."
            )
        if len(destination) > MAX_DESTINATION:
            return api_error(
                422,
                "invalid_destination",
                f"Destination must be at most {MAX_DESTINATION} characters.",
            )
        if len(category) > MAX_CATEGORY:
            return api_error(
                422,
                "invalid_category",
                f"Category must be at most {MAX_CATEGORY} characters.",
            )
        try:
            return await active.search_places(destination, category)
        except DiscoveryError as exc:
            return api_error(503, "discovery_unavailable", str(exc))

    return router
