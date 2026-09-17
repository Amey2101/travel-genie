from __future__ import annotations

from fastapi.testclient import TestClient

from app.agent.state import WorkflowStatus
from app.main import create_app
from app.providers.llm import LLMExtractionOutput
from tests.conftest import _make_agent_from_script


def _goa_only() -> LLMExtractionOutput:
    return LLMExtractionOutput(trip_destination="Goa")


def _complete() -> LLMExtractionOutput:
    return LLMExtractionOutput(
        trip_origin="Mumbai",
        trip_destination="Goa",
        duration_days=3,
        traveler_count=2,
    )


def _client_with(script: list[LLMExtractionOutput]) -> TestClient:
    agent, _calls = _make_agent_from_script(script)
    return TestClient(create_app(agent=agent))


class TestWorkflowAPI:
    def test_create_workflow(self) -> None:
        client = _client_with([_complete()])
        resp = client.post("/workflows")
        assert resp.status_code == 201
        assert "workflow_id" in resp.json()

    def test_full_flow_missing_then_complete(self) -> None:
        client = _client_with([_goa_only(), _complete()])
        wf_id = client.post("/workflows").json()["workflow_id"]
        first = client.post(f"/workflows/{wf_id}/messages", json={"text": "Goa trip"})
        assert first.status_code == 200
        assert first.json()["status"] == WorkflowStatus.AWAITING_CLARIFICATION

        second = client.post(
            f"/workflows/{wf_id}/messages",
            json={"text": "from Mumbai, 3 days, 2 people"},
        )
        assert second.status_code == 200
        assert second.json()["status"] == WorkflowStatus.REQUIREMENTS_COMPLETE
        assert second.json()["action"] == "search"
        assert second.json()["search_handoff"] is not None

    def test_get_workflow(self) -> None:
        client = _client_with([_goa_only()])
        wf_id = client.post("/workflows").json()["workflow_id"]
        get_resp = client.get(f"/workflows/{wf_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["status"] == WorkflowStatus.COLLECTING_REQUIREMENTS

    def test_get_unknown_workflow_404(self) -> None:
        client = _client_with([_complete()])
        resp = client.get("/workflows/nonexistent")
        assert resp.status_code == 404

    def test_post_message_unknown_workflow_404(self) -> None:
        client = _client_with([_complete()])
        resp = client.post(
            "/workflows/nonexistent/messages", json={"text": "hi"}
        )
        assert resp.status_code == 404

    def test_empty_text_422(self) -> None:
        client = _client_with([_complete()])
        wf_id = client.post("/workflows").json()["workflow_id"]
        resp = client.post(f"/workflows/{wf_id}/messages", json={"text": ""})
        assert resp.status_code == 422

    def test_health_still_works(self) -> None:
        client = _client_with([])
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
