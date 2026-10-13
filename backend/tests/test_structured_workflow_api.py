"""End-to-end API tests for the structured-input integration (Milestone 5)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.agent.orchestrator import TravelAgent
from app.agent.state import WorkflowStatus
from app.agent.store import InMemoryWorkflowStore
from app.decision.deterministic import DeterministicDecisionEngine
from app.main import create_app
from app.providers.demo_search import DemoSearchProvider
from app.providers.llm import LLMExtractionOutput
from app.providers.stub_llm import StubLLMClient
from app.services.requirement_extraction import RequirementExtractionService
from app.tools.search import SearchError, SearchRequest, SearchResponse

_ISOLATED_ENV_KEYS = (
    "LLM_PROVIDER",
    "SEARCH_PROVIDER",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "SERPAPI_API_KEY",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    for key in _ISOLATED_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


COMPLETE_PAYLOAD = {
    "origin": "Pune",
    "destination": "Goa",
    "departure_date": "2026-11-10",
    "return_date": "2026-11-12",
    "travelers": 2,
    "budget": 25000,
    "currency": "INR",
    "travel_style": "balanced",
    "interests": ["food", "beaches"],
}


class _FailingProvider:
    def search(self, request: SearchRequest) -> SearchResponse:
        raise SearchError("no live inventory configured")


def _scripted_client(script, *, provider=None) -> TestClient:
    calls: list = []

    def factory(request):
        index = len(calls)
        calls.append(request)
        return script[index] if index < len(script) else LLMExtractionOutput()

    service = RequirementExtractionService(
        StubLLMClient(factory=factory),
        reference_datetime=datetime(2026, 10, 13, tzinfo=timezone.utc),
    )
    agent = TravelAgent(
        extraction=service,
        store=InMemoryWorkflowStore(),
        search_provider=provider,
        decision_engine=DeterministicDecisionEngine() if provider else None,
    )
    return TestClient(create_app(agent=agent))


class TestStructuredRequestReachesPipeline:
    def test_complete_form_reaches_search_and_recommendation(self) -> None:
        client = TestClient(create_app())
        wf_id = client.post("/workflows").json()["workflow_id"]
        resp = client.post(
            f"/workflows/{wf_id}/requirements", json=COMPLETE_PAYLOAD
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == WorkflowStatus.REQUIREMENTS_COMPLETE
        assert body["action"] == "search"
        assert body["search_handoff"] is not None
        assert body["recommendation"]["status"] == "ok"
        assert body["recommendation"]["options"]

    def test_recommendation_is_persisted_in_workflow_state(self) -> None:
        client = TestClient(create_app())
        wf_id = client.post("/workflows").json()["workflow_id"]
        client.post(f"/workflows/{wf_id}/requirements", json=COMPLETE_PAYLOAD)
        state = client.get(f"/workflows/{wf_id}").json()
        assert state["recommendation"]["status"] == "ok"
        assert state["search_results"]

    def test_budget_is_a_target_not_a_hard_max(self) -> None:
        client = TestClient(create_app())
        wf_id = client.post("/workflows").json()["workflow_id"]
        body = client.post(
            f"/workflows/{wf_id}/requirements", json=COMPLETE_PAYLOAD
        ).json()
        req = body["requirements"]
        assert req["budget"]["is_approximate"] is True
        assert req["hard_constraints"]["max_budget"] is None


class TestDemoResultsAreLabelled:
    def test_options_are_demo_and_not_bookable(self) -> None:
        client = TestClient(create_app())
        wf_id = client.post("/workflows").json()["workflow_id"]
        body = client.post(
            f"/workflows/{wf_id}/requirements", json=COMPLETE_PAYLOAD
        ).json()
        options = body["recommendation"]["options"]
        assert options
        for option in options:
            assert option["candidate"]["source"] == "demo"
            assert option["candidate"]["bookable"] is False


class TestClarificationAndMultiTurn:
    def test_missing_timing_triggers_clarification(self) -> None:
        client = TestClient(create_app())
        wf_id = client.post("/workflows").json()["workflow_id"]
        payload = dict(COMPLETE_PAYLOAD)
        payload.pop("return_date")
        resp = client.post(f"/workflows/{wf_id}/requirements", json=payload)
        body = resp.json()
        assert body["status"] == WorkflowStatus.AWAITING_CLARIFICATION
        assert body["action"] == "ask_clarification"
        assert any(
            q["field"] == "travel.timing" for q in body["clarification_questions"]
        )
        assert body["recommendation"] is None

    def test_structured_then_text_turn_completes(self) -> None:
        complete = LLMExtractionOutput(
            trip_origin="Pune",
            trip_destination="Goa",
            duration_days=3,
            traveler_count=2,
        )
        client = _scripted_client([complete], provider=DemoSearchProvider())
        wf_id = client.post("/workflows").json()["workflow_id"]

        payload = dict(COMPLETE_PAYLOAD)
        payload.pop("return_date")
        first = client.post(f"/workflows/{wf_id}/requirements", json=payload).json()
        assert first["status"] == WorkflowStatus.AWAITING_CLARIFICATION

        second = client.post(
            f"/workflows/{wf_id}/messages",
            json={"text": "3 days, 2 people"},
        ).json()
        assert second["status"] == WorkflowStatus.REQUIREMENTS_COMPLETE
        assert second["recommendation"]["status"] == "ok"


class TestProviderFailureSurfaced:
    def test_provider_unavailable_is_explicit(self) -> None:
        client = _scripted_client([LLMExtractionOutput()], provider=_FailingProvider())
        wf_id = client.post("/workflows").json()["workflow_id"]
        body = client.post(
            f"/workflows/{wf_id}/requirements", json=COMPLETE_PAYLOAD
        ).json()
        assert body["status"] == WorkflowStatus.REQUIREMENTS_COMPLETE
        assert body["recommendation"]["status"] == "provider_unavailable"
        assert body["recommendation"]["options"] == []


class TestStructuredValidation:
    def test_unknown_interest_returns_422_envelope(self) -> None:
        client = TestClient(create_app())
        wf_id = client.post("/workflows").json()["workflow_id"]
        payload = dict(COMPLETE_PAYLOAD, interests=["skiing"])
        resp = client.post(f"/workflows/{wf_id}/requirements", json=payload)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"

    def test_return_before_departure_returns_422(self) -> None:
        client = TestClient(create_app())
        wf_id = client.post("/workflows").json()["workflow_id"]
        payload = dict(
            COMPLETE_PAYLOAD,
            departure_date="2026-11-10",
            return_date="2026-11-01",
        )
        resp = client.post(f"/workflows/{wf_id}/requirements", json=payload)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "validation_error"

    def test_return_and_duration_mismatch_returns_422(self) -> None:
        client = TestClient(create_app())
        wf_id = client.post("/workflows").json()["workflow_id"]
        payload = dict(
            COMPLETE_PAYLOAD, return_date="2026-11-12", duration_days=9
        )
        resp = client.post(f"/workflows/{wf_id}/requirements", json=payload)
        assert resp.status_code == 422

    def test_unknown_workflow_returns_404(self) -> None:
        client = TestClient(create_app())
        resp = client.post(
            "/workflows/nonexistent/requirements", json=COMPLETE_PAYLOAD
        )
        assert resp.status_code == 404


class TestDeterministicClarificationAnswer:
    """The default (stub) provider cannot parse free text, so clarifications
    must be answerable deterministically through the structured endpoint."""

    def _start_missing_timing(self, client: TestClient) -> str:
        wf_id = client.post("/workflows").json()["workflow_id"]
        payload = dict(COMPLETE_PAYLOAD)
        payload.pop("return_date")
        first = client.post(
            f"/workflows/{wf_id}/requirements", json=payload
        ).json()
        assert first["status"] == WorkflowStatus.AWAITING_CLARIFICATION
        return wf_id

    def test_duration_answer_completes_without_llm(self) -> None:
        client = TestClient(create_app())
        wf_id = self._start_missing_timing(client)
        body = client.post(
            f"/workflows/{wf_id}/clarifications", json={"duration_days": 3}
        ).json()
        assert body["status"] == WorkflowStatus.REQUIREMENTS_COMPLETE
        assert body["recommendation"]["status"] == "ok"
        req = body["requirements"]
        assert req["trip"]["origin"] == "Pune"
        assert req["trip"]["destination"] == "Goa"
        assert req["budget"]["is_approximate"] is True

    def test_return_date_answer_completes(self) -> None:
        client = TestClient(create_app())
        wf_id = self._start_missing_timing(client)
        body = client.post(
            f"/workflows/{wf_id}/clarifications",
            json={"departure_date": "2026-11-10", "return_date": "2026-11-12"},
        ).json()
        assert body["status"] == WorkflowStatus.REQUIREMENTS_COMPLETE
        assert body["recommendation"]["status"] == "ok"

    def test_answer_that_omits_timing_stays_awaiting(self) -> None:
        client = TestClient(create_app())
        wf_id = self._start_missing_timing(client)
        body = client.post(
            f"/workflows/{wf_id}/clarifications", json={"travelers": 4}
        ).json()
        assert body["status"] == WorkflowStatus.AWAITING_CLARIFICATION
        assert any(
            q["field"] == "travel.timing" for q in body["clarification_questions"]
        )

    def test_unknown_workflow_returns_404(self) -> None:
        client = TestClient(create_app())
        resp = client.post(
            "/workflows/nonexistent/clarifications", json={"duration_days": 3}
        )
        assert resp.status_code == 404

    def test_no_prior_requirements_returns_409(self) -> None:
        client = TestClient(create_app())
        wf_id = client.post("/workflows").json()["workflow_id"]
        resp = client.post(
            f"/workflows/{wf_id}/clarifications", json={"duration_days": 3}
        )
        assert resp.status_code == 409

    def test_inconsistent_answer_returns_422(self) -> None:
        client = TestClient(create_app())
        wf_id = self._start_missing_timing(client)
        resp = client.post(
            f"/workflows/{wf_id}/clarifications",
            json={"return_date": "2026-11-12"},
        )
        assert resp.status_code == 422
