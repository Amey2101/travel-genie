"""Frontend/backend contract tests for the Milestone 5 workflow integration.

These are static-asset assertions: the project has no JavaScript test runner, so
we verify the served JavaScript targets the real workflow endpoints (and the
real response field names) rather than executing it in a browser. Browser-level
behaviour remains untested; see the milestone report.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    for key in ("GEMINI_API_KEY", "GROQ_API_KEY", "SERPAPI_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestHomePageUsesWorkflow:
    def test_app_js_targets_workflow_endpoints(self, client: TestClient) -> None:
        js = client.get("/static/app.js").text
        assert 'fetch("/workflows", { method: "POST" })' in js
        assert 'encodeURIComponent(workflowId) + "/requirements"' in js
        assert 'encodeURIComponent(workflowId) + "/clarifications"' in js
        assert '"/api/health"' in js

    def test_app_js_no_longer_submits_to_plan(self, client: TestClient) -> None:
        js = client.get("/static/app.js").text
        assert '"/api/plan"' not in js
        assert 'fetch("/api/plan"' not in js

    def test_app_js_renders_backend_recommendation_contract(
        self, client: TestClient
    ) -> None:
        js = client.get("/static/app.js").text
        for field in (
            "rec.options",
            "rec.rejected",
            "rec.status",
            "opt.candidate",
            "c.bookable",
            "c.source",
            "opt.reasons",
            "body.clarification_questions",
            "body.requirements",
        ):
            assert field in js

    def test_app_js_labels_demo_results_non_bookable(
        self, client: TestClient
    ) -> None:
        js = client.get("/static/app.js").text
        assert "Demo · not bookable" in js
        assert "not bookable" in js

    def test_home_copy_does_not_promise_itinerary_or_booking(
        self, client: TestClient
    ) -> None:
        html = client.get("/").text
        assert "day-by-day itinerary" not in html
        assert "Nothing is booked" in html

    def test_plan_placeholder_retained_for_compatibility(
        self, client: TestClient
    ) -> None:
        resp = client.post("/api/plan", json={})
        assert resp.status_code == 501
        assert resp.json()["error"]["code"] == "not_implemented"


class TestDiscoveryRemainsIntact:
    def test_discover_page_and_script_still_present(self, client: TestClient) -> None:
        assert client.get("/discover").status_code == 200
        js = client.get("/static/discover.js").text
        assert "/api/discover" in js

    def test_discover_route_registered(self, client: TestClient) -> None:
        assert "/api/discover" in app.openapi()["paths"]
