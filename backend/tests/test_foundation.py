"""Foundational application-integration tests (Milestone 1)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

SECRET_GEMINI = "AIzaSyFAKE_foundation_test_key_0000000000"
SECRET_SERP = "serp-secret-foundation-key"


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    """Start every test in demo mode, regardless of the developer's real .env."""
    for key in ("GEMINI_API_KEY", "GROQ_API_KEY", "SERPAPI_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestHealth:
    def test_plain_health_is_unchanged(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_api_health_reports_demo_mode_without_keys(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok" and body["mode"] == "demo"
        assert body["integrations"]["gemini"]["configured"] is False
        assert body["integrations"]["groq"]["configured"] is False
        assert body["integrations"]["serpapi"]["configured"] is False

    def test_api_health_never_leaks_keys(self, client: TestClient, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        monkeypatch.setenv("SERPAPI_API_KEY", SECRET_SERP)
        resp = client.get("/api/health")
        assert resp.json()["integrations"]["gemini"]["configured"] is True
        assert resp.json()["integrations"]["serpapi"]["configured"] is True
        assert resp.json()["mode"] == "live"
        assert SECRET_GEMINI not in resp.text
        assert SECRET_SERP not in resp.text


class TestStaticFrontend:
    def test_index_page_is_served(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Travel-Genie" in resp.text

    def test_discover_page_is_served(self, client: TestClient) -> None:
        resp = client.get("/discover")
        assert resp.status_code == 200
        assert "Explore places" in resp.text

    def test_static_assets_are_served(self, client: TestClient) -> None:
        js = client.get("/static/app.js")
        assert js.status_code == 200
        assert '"/api/health"' in js.text
        assert client.get("/static/styles.css").status_code == 200
        assert client.get("/static/discover.js").status_code == 200


class TestErrorEnvelope:
    def test_plan_compat_is_not_implemented(self, client: TestClient) -> None:
        resp = client.post("/api/plan", json={})
        assert resp.status_code == 501
        error = resp.json()["error"]
        assert error["code"] == "not_implemented"
        assert error["details"] == []

    def test_unknown_route_uses_envelope(self, client: TestClient) -> None:
        resp = client.get("/api/nothing")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "http_error"

    def test_validation_error_is_wrapped(self, client: TestClient) -> None:
        wf_id = client.post("/workflows").json()["workflow_id"]
        resp = client.post(f"/workflows/{wf_id}/messages", json={"text": ""})
        assert resp.status_code == 422
        error = resp.json()["error"]
        assert error["code"] == "validation_error"
        assert error["details"]

    def test_malformed_json_is_wrapped(self, client: TestClient) -> None:
        wf_id = client.post("/workflows").json()["workflow_id"]
        resp = client.post(
            f"/workflows/{wf_id}/messages",
            content="{nope",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"
