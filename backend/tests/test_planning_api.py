"""Workflow API tests for the planning pipeline (Milestone 4)."""

from __future__ import annotations

from datetime import datetime, timezone

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
from app.tools.search import SearchError, SearchResponse, SearchRequest


class _FailingProvider:
    def search(self, request: SearchRequest) -> SearchResponse:
        raise SearchError("no live inventory configured")


class _EmptyProvider:
    def search(self, request: SearchRequest) -> SearchResponse:
        return SearchResponse(workflow_id=request.workflow_id, results=[])


def _complete() -> LLMExtractionOutput:
    return LLMExtractionOutput(
        trip_origin="Mumbai",
        trip_destination="Goa",
        duration_days=3,
        traveler_count=2,
    )


def _client(provider) -> TestClient:
    calls: list = []

    def factory(request):
        index = len(calls)
        calls.append(request)
        return _complete()

    client = StubLLMClient(factory=factory)
    service = RequirementExtractionService(
        client, reference_datetime=datetime(2026, 10, 13, tzinfo=timezone.utc)
    )
    agent = TravelAgent(
        extraction=service,
        store=InMemoryWorkflowStore(),
        search_provider=provider,
        decision_engine=DeterministicDecisionEngine(),
    )
    return TestClient(create_app(agent=agent))


class TestPlanningAPI:
    def test_complete_message_returns_recommendation(self) -> None:
        client = _client(DemoSearchProvider())
        wf_id = client.post("/workflows").json()["workflow_id"]
        resp = client.post(
            f"/workflows/{wf_id}/messages",
            json={"text": "Mumbai to Goa 3 days 2 people"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == WorkflowStatus.REQUIREMENTS_COMPLETE
        assert body["action"] == "search"
        assert body["search_handoff"] is not None
        assert body["recommendation"]["status"] == "ok"
        assert body["recommendation"]["options"]
        assert all(
            option["candidate"]["source"] == "demo"
            and option["candidate"]["bookable"] is False
            for option in body["recommendation"]["options"]
        )

    def test_get_workflow_exposes_persisted_planning(self) -> None:
        client = _client(DemoSearchProvider())
        wf_id = client.post("/workflows").json()["workflow_id"]
        client.post(
            f"/workflows/{wf_id}/messages",
            json={"text": "Mumbai to Goa 3 days 2 people"},
        )
        state = client.get(f"/workflows/{wf_id}").json()
        assert state["search_results"]
        assert state["recommendation"]["status"] == "ok"

    def test_provider_failure_is_not_a_500(self) -> None:
        client = _client(_FailingProvider())
        wf_id = client.post("/workflows").json()["workflow_id"]
        resp = client.post(
            f"/workflows/{wf_id}/messages",
            json={"text": "Mumbai to Goa 3 days 2 people"},
        )
        assert resp.status_code == 200
        assert resp.json()["recommendation"]["status"] == "provider_unavailable"

    def test_empty_results_reported(self) -> None:
        client = _client(_EmptyProvider())
        wf_id = client.post("/workflows").json()["workflow_id"]
        resp = client.post(
            f"/workflows/{wf_id}/messages",
            json={"text": "Mumbai to Goa 3 days 2 people"},
        )
        assert resp.status_code == 200
        assert resp.json()["recommendation"]["status"] == "no_results"


class TestUnchangedEndpoints:
    def test_plan_placeholder_unchanged(self) -> None:
        client = _client(DemoSearchProvider())
        resp = client.post("/api/plan", json={})
        assert resp.status_code == 501
        assert resp.json()["error"]["code"] == "not_implemented"

    def test_health_unchanged(self) -> None:
        client = _client(DemoSearchProvider())
        assert client.get("/health").json() == {"status": "ok"}

    def test_api_health_unchanged_shape(self) -> None:
        client = _client(DemoSearchProvider())
        body = client.get("/api/health").json()
        assert {"status", "mode", "env_file_found", "integrations"} <= set(body)


class TestDefaultAppWiring:
    def test_default_agent_plans_end_to_end(self) -> None:
        client = TestClient(create_app())
        wf_id = client.post("/workflows").json()["workflow_id"]
        resp = client.post(
            f"/workflows/{wf_id}/messages",
            json={"text": "Plan a trip"},
        )
        # Default extraction stub yields no facts -> clarification, no crash.
        assert resp.status_code == 200
        assert resp.json()["status"] == WorkflowStatus.AWAITING_CLARIFICATION
