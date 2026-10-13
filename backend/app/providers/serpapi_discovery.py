"""Concrete destination-discovery adapter backed by SerpApi Google Maps.

Ported and adapted from the teammate implementation's
``services/discovery.py``. Discovery answers "what is there to see and do",
not "what can I book": results are places/attractions with no bookable price or
availability. It is intentionally *not* a ``SearchProvider`` (inventory).

Safety properties:
- The API key is read server-side from config and never logged or returned.
- Provider/network errors are normalized to fixed, credential-free messages.
- Only ordinary ``http(s)`` links survive normalization; ``javascript:`` and
  ``data:`` URLs are dropped.
"""

from __future__ import annotations

from urllib.parse import quote_plus, urlparse

import httpx

from app.core import config
from app.tools.discovery import (
    DiscoveryError,
    DiscoveryResponse,
    DiscoveryResult,
)

SERPAPI_URL = "https://serpapi.com/search.json"
MAX_RESULTS = 18


def _safe_http_url(value: object) -> str:
    """Return only ordinary HTTP(S) links; drop ``javascript:``/``data:`` etc."""
    if not isinstance(value, str):
        return ""
    parsed = urlparse(value)
    return value if parsed.scheme in ("http", "https") and parsed.netloc else ""


def _text(value: object, limit: int) -> str:
    return str(value).strip()[:limit] if value is not None else ""


def _maps_url(item: dict, query: str, destination: str) -> str:
    place_id = item.get("place_id")
    if isinstance(place_id, str) and place_id:
        return "https://www.google.com/maps/place/?q=place_id:" + quote_plus(place_id)
    return "https://www.google.com/maps/search/?api=1&query=" + quote_plus(
        f"{item.get('title') or query} {destination} India"
    )


def normalize_result(
    item: dict, query: str, destination: str
) -> DiscoveryResult | None:
    """Normalize a Google Maps local result without assuming optional fields."""
    title = _text(item.get("title") or item.get("name"), 240)
    if not title:
        return None
    website = _safe_http_url(item.get("website", ""))
    thumbnail = _safe_http_url(item.get("thumbnail", ""))
    links = item.get("links") if isinstance(item.get("links"), dict) else {}
    provider_link = _safe_http_url(links.get("website", ""))
    rating = item.get("rating")
    reviews = item.get("reviews")
    return DiscoveryResult(
        title=title,
        category=_text(item.get("type") or item.get("category") or "Place", 100),
        address=_text(item.get("address"), 400),
        rating=rating if isinstance(rating, (int, float)) else None,
        reviews=reviews if isinstance(reviews, int) else None,
        price=_text(item.get("price"), 80),
        description=_text(item.get("description") or item.get("snippet"), 500),
        hours=_text(item.get("hours"), 160),
        thumbnail=thumbnail,
        website=website or provider_link,
        maps_url=_maps_url(item, query, destination),
    )


class SerpApiDiscoveryProvider:
    """``DiscoveryProvider`` implementation over SerpApi Google Maps."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float | None = None,
    ) -> None:
        self._client = client
        self._transport = transport
        self._timeout = timeout if timeout is not None else config.serpapi_timeout_seconds()

    async def search_places(
        self, destination: str, category: str
    ) -> DiscoveryResponse:
        key = config.serpapi_key()
        if not key:
            raise DiscoveryError(
                "Live discovery is unavailable: no SERPAPI_API_KEY is configured "
                "on the server."
            )

        destination = " ".join(destination.split())[:100]
        category = " ".join(category.split())[:80] or "tourist attractions"
        if not destination:
            raise DiscoveryError("Enter a destination to search.")

        query = f"{category} in {destination}, India"
        own_client = self._client is None
        client = self._client or httpx.AsyncClient(
            transport=self._transport, timeout=self._timeout
        )
        try:
            try:
                response = await client.get(
                    SERPAPI_URL,
                    params={
                        "engine": "google_maps",
                        "q": query,
                        "type": "search",
                        "hl": "en",
                        "gl": "in",
                        "api_key": key,
                    },
                )
            except httpx.HTTPError as exc:
                # httpx messages can embed the request URL (and therefore the
                # key), so never surface them.
                raise DiscoveryError(
                    "Could not reach the discovery provider. Please try again."
                ) from exc

            if response.status_code >= 400:
                raise DiscoveryError(
                    f"The discovery provider rejected the request (HTTP "
                    f"{response.status_code}). Check the API key, quota and plan."
                )
            try:
                data = response.json()
            except ValueError as exc:
                raise DiscoveryError(
                    "The discovery provider returned an unreadable response."
                ) from exc
            if not isinstance(data, dict):
                raise DiscoveryError(
                    "The discovery provider returned an unexpected response format."
                )
            raw_results = data.get("local_results") or data.get("places_results") or []
            if (
                data.get("error")
                and not data.get("local_results")
                and not data.get("places_results")
            ):
                raise DiscoveryError(
                    "The discovery provider could not complete this search. "
                    "Check the API quota and try a simpler query."
                )
            return self._to_response(raw_results, destination, category, query)
        finally:
            if own_client:
                await client.aclose()

    def _to_response(
        self, raw_results: object, destination: str, category: str, query: str
    ) -> DiscoveryResponse:
        cards: list[DiscoveryResult] = []
        if isinstance(raw_results, list):
            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                normalized = normalize_result(item, query, destination)
                if normalized is not None:
                    cards.append(normalized)
                if len(cards) >= MAX_RESULTS:
                    break
        return DiscoveryResponse(
            destination=destination,
            category=category,
            query=query,
            count=len(cards),
            results=cards,
        )
