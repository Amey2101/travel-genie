"""Focused tests for the SerpApi destination-discovery integration (Milestone 2).

All HTTP is mocked with ``httpx.MockTransport``; no live SerpApi calls are made.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.providers.serpapi_discovery import (
    SerpApiDiscoveryProvider,
    normalize_result,
)
from app.tools.discovery import DiscoveryError
from app.tools.search import SearchProvider

SECRET = "serpapi-secret-key-do-not-leak-123"


def _provider(handler) -> SerpApiDiscoveryProvider:
    return SerpApiDiscoveryProvider(transport=httpx.MockTransport(handler))


def _client(provider) -> TestClient:
    return TestClient(create_app(discovery_provider=provider))


def _ok_handler(seen: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={
                "local_results": [
                    {
                        "title": "Hawa Mahal",
                        "type": "Tourist attraction",
                        "rating": 4.5,
                        "reviews": 1200,
                        "address": "Badi Choupad, Jaipur",
                        "place_id": "xyz123",
                        "website": "https://example.com/hawa",
                        "thumbnail": "https://example.com/hawa.jpg",
                    },
                    {"type": "Missing title"},
                ]
            },
        )

    return handler


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


class TestNormalization:
    def test_handles_optional_fields(self) -> None:
        result = normalize_result(
            {
                "title": "Amber Fort",
                "type": "Historical place",
                "rating": 4.7,
                "reviews": 1200,
                "address": "Devisinghpura, Jaipur",
                "place_id": "abc123",
                "thumbnail": "https://example.com/amber.jpg",
                "website": "https://example.com/fort",
            },
            "forts",
            "Jaipur",
        )
        assert result is not None
        assert result.title == "Amber Fort"
        assert result.rating == 4.7 and result.reviews == 1200
        assert result.maps_url.endswith("abc123")
        assert result.website == "https://example.com/fort"

    def test_rejects_unsafe_links(self) -> None:
        result = normalize_result(
            {
                "title": "Example",
                "website": "javascript:alert(1)",
                "thumbnail": "data:text/html,<script>x</script>",
            },
            "places",
            "Delhi",
        )
        assert result is not None
        assert result.website == ""
        assert result.thumbnail == ""
        assert result.maps_url.startswith("https://www.google.com/maps/")

    def test_missing_title_is_dropped(self) -> None:
        assert normalize_result({"type": "Attraction"}, "places", "Delhi") is None

    def test_discovery_result_is_not_inventory(self) -> None:
        result = normalize_result({"title": "X", "place_id": "p1"}, "places", "Delhi")
        payload = result.model_dump()
        assert "provider_ref" not in payload
        assert "currency" not in payload
        assert "availability" not in payload


# --------------------------------------------------------------------------
# Provider (direct, mocked HTTP)
# --------------------------------------------------------------------------


class TestProvider:
    def test_uses_google_maps_india_params(self, monkeypatch) -> None:
        monkeypatch.setenv("SERPAPI_API_KEY", SECRET)
        seen: dict = {}
        provider = _provider(_ok_handler(seen))
        response = asyncio.run(
            provider.search_places("  Jaipur ", "tourist attractions")
        )
        assert response.count == 1
        assert response.destination == "Jaipur"
        assert response.results[0].title == "Hawa Mahal"
        assert seen["params"]["engine"] == "google_maps"
        assert seen["params"]["gl"] == "in"
        assert seen["params"]["q"] == "tourist attractions in Jaipur, India"
        assert SECRET not in response.model_dump_json()

    def test_missing_credentials_raise(self, monkeypatch) -> None:
        monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
        provider = _provider(_ok_handler())
        with pytest.raises(DiscoveryError) as exc:
            asyncio.run(provider.search_places("Delhi", "hotels"))
        assert "SERPAPI_API_KEY" in str(exc.value)

    def test_timeout_is_sanitized(self, monkeypatch) -> None:
        monkeypatch.setenv("SERPAPI_API_KEY", SECRET)

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("boom", request=request)

        with pytest.raises(DiscoveryError) as exc:
            asyncio.run(_provider(handler).search_places("Delhi", "hotels"))
        assert SECRET not in str(exc.value)
        assert "key" not in str(exc.value).lower()

    def test_http_error_is_sanitized(self, monkeypatch) -> None:
        monkeypatch.setenv("SERPAPI_API_KEY", SECRET)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text=f"server blew up api_key={SECRET}")

        with pytest.raises(DiscoveryError) as exc:
            asyncio.run(_provider(handler).search_places("Delhi", "hotels"))
        assert SECRET not in str(exc.value)

    def test_provider_error_payload_is_not_echoed(self, monkeypatch) -> None:
        monkeypatch.setenv("SERPAPI_API_KEY", SECRET)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"error": f"Invalid API key: {SECRET}"})

        with pytest.raises(DiscoveryError) as exc:
            asyncio.run(_provider(handler).search_places("Delhi", "hotels"))
        assert SECRET not in str(exc.value)

    def test_unreadable_response_is_handled(self, monkeypatch) -> None:
        monkeypatch.setenv("SERPAPI_API_KEY", SECRET)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>not json</html>")

        with pytest.raises(DiscoveryError):
            asyncio.run(_provider(handler).search_places("Delhi", "hotels"))


# --------------------------------------------------------------------------
# API: GET /api/discover
# --------------------------------------------------------------------------


class TestDiscoveryAPI:
    def test_success_returns_consistent_shape(self, monkeypatch) -> None:
        monkeypatch.setenv("SERPAPI_API_KEY", SECRET)
        resp = _client(_provider(_ok_handler())).get(
            "/api/discover",
            params={"destination": "Jaipur", "category": "tourist attractions"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {"destination", "category", "query", "count", "results"}
        assert body["count"] == 1
        assert body["results"][0]["title"] == "Hawa Mahal"
        assert body["results"][0]["maps_url"].startswith("https://www.google.com/maps/")
        assert SECRET not in resp.text

    def test_default_category(self, monkeypatch) -> None:
        monkeypatch.setenv("SERPAPI_API_KEY", SECRET)
        resp = _client(_provider(_ok_handler())).get(
            "/api/discover", params={"destination": "Jaipur"}
        )
        assert resp.status_code == 200
        assert resp.json()["category"] == "tourist attractions"

    def test_missing_destination_is_422(self, monkeypatch) -> None:
        monkeypatch.setenv("SERPAPI_API_KEY", SECRET)
        resp = _client(_provider(_ok_handler())).get("/api/discover")
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "invalid_destination"

    def test_blank_destination_is_422(self, monkeypatch) -> None:
        monkeypatch.setenv("SERPAPI_API_KEY", SECRET)
        resp = _client(_provider(_ok_handler())).get(
            "/api/discover", params={"destination": "   "}
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "invalid_destination"

    def test_destination_too_long_is_422(self, monkeypatch) -> None:
        monkeypatch.setenv("SERPAPI_API_KEY", SECRET)
        resp = _client(_provider(_ok_handler())).get(
            "/api/discover", params={"destination": "x" * 101}
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "invalid_destination"

    def test_category_too_long_is_422(self, monkeypatch) -> None:
        monkeypatch.setenv("SERPAPI_API_KEY", SECRET)
        resp = _client(_provider(_ok_handler())).get(
            "/api/discover",
            params={"destination": "Jaipur", "category": "y" * 81},
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "invalid_category"

    def test_missing_credentials_is_503(self, monkeypatch) -> None:
        monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
        resp = _client(_provider(_ok_handler())).get(
            "/api/discover", params={"destination": "Jaipur"}
        )
        assert resp.status_code == 503
        error = resp.json()["error"]
        assert error["code"] == "discovery_unavailable"
        assert "SERPAPI_API_KEY" in error["message"]
        assert SECRET not in resp.text

    def test_timeout_is_503(self, monkeypatch) -> None:
        monkeypatch.setenv("SERPAPI_API_KEY", SECRET)

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("boom", request=request)

        resp = _client(_provider(handler)).get(
            "/api/discover", params={"destination": "Jaipur"}
        )
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "discovery_unavailable"

    def test_provider_http_error_is_503(self, monkeypatch) -> None:
        monkeypatch.setenv("SERPAPI_API_KEY", SECRET)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text=f"quota exceeded key={SECRET}")

        resp = _client(_provider(handler)).get(
            "/api/discover", params={"destination": "Jaipur"}
        )
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "discovery_unavailable"
        assert SECRET not in resp.text

    def test_provider_error_payload_is_503_and_sanitized(self, monkeypatch) -> None:
        monkeypatch.setenv("SERPAPI_API_KEY", SECRET)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"error": f"bad key {SECRET}"})

        resp = _client(_provider(handler)).get(
            "/api/discover", params={"destination": "Jaipur"}
        )
        assert resp.status_code == 503
        assert SECRET not in resp.text

    def test_unsafe_result_urls_are_dropped(self, monkeypatch) -> None:
        monkeypatch.setenv("SERPAPI_API_KEY", SECRET)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "local_results": [
                        {
                            "title": "Sketchy Place",
                            "website": "javascript:alert(1)",
                            "thumbnail": "data:image/png;base64,AAAA",
                        }
                    ]
                },
            )

        resp = _client(_provider(handler)).get(
            "/api/discover", params={"destination": "Jaipur"}
        )
        assert resp.status_code == 200
        item = resp.json()["results"][0]
        assert item["website"] == ""
        assert item["thumbnail"] == ""
        assert item["maps_url"].startswith("https://www.google.com/maps/")

    def test_no_results_is_200_empty(self, monkeypatch) -> None:
        monkeypatch.setenv("SERPAPI_API_KEY", SECRET)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"local_results": []})

        resp = _client(_provider(handler)).get(
            "/api/discover", params={"destination": "Jaipur"}
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 0
        assert resp.json()["results"] == []


class TestArchitecturalSeparation:
    def test_discovery_provider_is_not_inventory_search_provider(self) -> None:
        provider = _provider(lambda request: httpx.Response(200, json={}))
        assert not isinstance(provider, SearchProvider)
