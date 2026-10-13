"""Tests for the live LLM providers and provider selection (Milestone 3).

All HTTP is mocked with ``httpx.MockTransport``; no live provider calls are
made. Missing credentials are tested without secrets; every credential-bearing
test uses a fixed fake key that must never appear in errors or logs.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import httpx
import pytest
from pydantic import ValidationError

from app.agent.orchestrator import TravelAgent
from app.agent.state import WorkflowStatus
from app.agent.store import InMemoryWorkflowStore
from app.core import config
from app.providers.factory import build_llm_client
from app.providers.gemini_llm import GeminiLLMClient
from app.providers.groq_llm import GroqLLMClient
from app.providers.llm import (
    LLMClientError,
    LLMExtractionOutput,
    LLMExtractionRequest,
)
from app.providers.stub_llm import StubLLMClient
from app.services.requirement_extraction import (
    InconsistentExtractionError,
    LLMOutputValidationError,
    LLMProviderError,
    RequirementExtractionService,
)

FIXED_DT = datetime(2026, 9, 3, 9, 0, tzinfo=timezone.utc)
SECRET_GEMINI = "AIzaSy_test_secret_gemini_key_0000000"
SECRET_GROQ = "gsk_test_secret_groq_key_0000000000"


def _request() -> LLMExtractionRequest:
    return LLMExtractionRequest(
        system_prompt="Extract travel requirements.",
        user_message="Mumbai to Goa",
        reference_datetime=FIXED_DT,
    )


def _json_output(**overrides) -> dict:
    data = {
        "trip_origin": "Mumbai",
        "trip_destination": "Goa",
        "trip_start_date": "2026-12-20",
        "trip_end_date": "2026-12-24",
        "duration_days": 4,
        "traveler_count": 2,
        "budget": {"amount": "30000", "currency": "INR", "kind": "approximate"},
    }
    data.update(overrides)
    return data


def _gemini_ok(json_text: str = "") -> GeminiLLMClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": json_text or json.dumps(_json_output())}]}}
                ]
            },
        )

    return GeminiLLMClient(transport=httpx.MockTransport(handler))


def _groq_ok(json_text: str = "") -> GroqLLMClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json_text or json.dumps(_json_output())}}
                ]
            },
        )

    return GroqLLMClient(transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------------
# Provider adapters — request construction & success paths
# --------------------------------------------------------------------------


class TestGeminiRequestAndParsing:
    def test_request_construction(self, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen.update(
                {
                    "url": str(request.url),
                    "key_header": request.headers.get("x-goog-api-key"),
                    "system": body["systemInstruction"]["parts"][0]["text"],
                    "user": body["contents"][0]["parts"][0]["text"],
                    "mime": body["generationConfig"]["responseMimeType"],
                    "temperature": body["generationConfig"]["temperature"],
                    "max_tokens": body["generationConfig"]["maxOutputTokens"],
                }
            )
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(_json_output())}]}}]})

        client = GeminiLLMClient(transport=httpx.MockTransport(handler))
        result = client.generate_structured(request=_request(), schema=LLMExtractionOutput)

        assert "models/gemini-3.8-flash:generateContent" in seen["url"]
        assert seen["key_header"] == SECRET_GEMINI
        assert "JSON Schema" in seen["system"]
        assert seen["user"] == "Mumbai to Goa"
        assert seen["mime"] == "application/json"
        assert seen["temperature"] == 0
        assert seen["max_tokens"] == config.gemini_max_output_tokens()
        assert result.trip_destination == "Goa"
        assert result.traveler_count == 2

    def test_model_selection_via_config(self, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(_json_output())}]}}]})

        client = GeminiLLMClient(transport=httpx.MockTransport(handler))
        client.generate_structured(request=_request(), schema=LLMExtractionOutput)
        # captured through handler below
        assert config.gemini_model() == "gemini-2.5-flash"

    def test_explicit_model_override(self, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(_json_output())}]}}]})

        client = GeminiLLMClient(transport=httpx.MockTransport(handler), model="gemini-flash-latest")
        client.generate_structured(request=_request(), schema=LLMExtractionOutput)
        assert "models/gemini-flash-latest:generateContent" in seen[0]

    def test_strips_markdown_fences(self, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        client = _gemini_ok("```json\n" + json.dumps(_json_output()) + "\n```")
        result = client.generate_structured(request=_request(), schema=LLMExtractionOutput)
        assert result.trip_destination == "Goa"

    def test_propagates_validation_failures(self, monkeypatch) -> None:
        """Deterministic validation stays authoritative: invalid JSON is rejected."""
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        client = _gemini_ok(json.dumps({"duration_days": 0}))
        with pytest.raises(ValidationError):
            client.generate_structured(request=_request(), schema=LLMExtractionOutput)


class TestGroqRequestAndParsing:
    def test_request_construction(self, monkeypatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", SECRET_GROQ)
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen.update(
                {
                    "url": str(request.url),
                    "auth": request.headers.get("authorization"),
                    "model": body["model"],
                    "system": body["messages"][0]["content"],
                    "user": body["messages"][1]["content"],
                    "response_format": body["response_format"],
                    "max_tokens": body["max_completion_tokens"],
                }
            )
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(_json_output())}}]})

        client = GroqLLMClient(transport=httpx.MockTransport(handler))
        result = client.generate_structured(request=_request(), schema=LLMExtractionOutput)

        assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
        assert seen["auth"] == f"Bearer {SECRET_GROQ}"
        assert seen["model"] == config.groq_model()
        assert "JSON Schema" in seen["system"]
        assert seen["user"] == "Mumbai to Goa"
        assert seen["response_format"] == {"type": "json_object"}
        assert seen["max_tokens"] == config.groq_max_output_tokens()
        assert result.trip_destination == "Goa"
        assert result.traveler_count == 2

    def test_model_selection_via_config_and_override(self, monkeypatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", SECRET_GROQ)
        monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content)["model"])
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(_json_output())}}]})

        client = GroqLLMClient(transport=httpx.MockTransport(handler))
        client.generate_structured(request=_request(), schema=LLMExtractionOutput)
        assert seen[-1] == "llama-3.3-70b-versatile"

        client_override = GroqLLMClient(transport=httpx.MockTransport(handler), model="openai/gpt-oss-20b")
        client_override.generate_structured(request=_request(), schema=LLMExtractionOutput)
        assert seen[-1] == "openai/gpt-oss-20b"

    def test_propagates_validation_failures(self, monkeypatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", SECRET_GROQ)
        client = _groq_ok(json.dumps({"traveler_count": -1}))
        with pytest.raises(ValidationError):
            client.generate_structured(request=_request(), schema=LLMExtractionOutput)


# --------------------------------------------------------------------------
# Provider adapters — failure modes
# --------------------------------------------------------------------------


class TestProviderFailureModes:
    @pytest.mark.parametrize(
        "provider_cls, key_var, secret",
        [
            (GeminiLLMClient, "GEMINI_API_KEY", SECRET_GEMINI),
            (GroqLLMClient, "GROQ_API_KEY", SECRET_GROQ),
        ],
    )
    def test_missing_key_is_controlled_and_actionable(
        self, monkeypatch, caplog, provider_cls, key_var, secret
    ) -> None:
        monkeypatch.delenv(key_var, raising=False)
        provider = provider_cls(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="{}")))
        with pytest.raises(LLMClientError) as exc:
            provider.generate_structured(request=_request(), schema=LLMExtractionOutput)
        assert key_var in str(exc.value)
        assert secret not in str(exc.value)
        assert secret not in caplog.text

    @pytest.mark.parametrize(
        "provider_cls, key_var, secret",
        [
            (GeminiLLMClient, "GEMINI_API_KEY", SECRET_GEMINI),
            (GroqLLMClient, "GROQ_API_KEY", SECRET_GROQ),
        ],
    )
    def test_auth_failure_sanitized(self, monkeypatch, caplog, provider_cls, key_var, secret) -> None:
        monkeypatch.setenv(key_var, secret)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text=f"Invalid credentials api_key={secret} bearer={secret}")

        provider = provider_cls(transport=httpx.MockTransport(handler))
        with pytest.raises(LLMClientError) as exc:
            provider.generate_structured(request=_request(), schema=LLMExtractionOutput)
        assert str(exc.value).startswith("The ")
        assert secret not in str(exc.value)
        assert secret not in caplog.text

    @pytest.mark.parametrize("provider_cls", [GeminiLLMClient, GroqLLMClient])
    def test_rate_limit_sanitized(self, monkeypatch, provider_cls) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        monkeypatch.setenv("GROQ_API_KEY", SECRET_GROQ)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text=f"quota exceeded key={SECRET_GEMINI}")

        provider = provider_cls(transport=httpx.MockTransport(handler))
        with pytest.raises(LLMClientError) as exc:
            provider.generate_structured(request=_request(), schema=LLMExtractionOutput)
        assert "rate limit" in str(exc.value).lower()
        assert SECRET_GEMINI not in str(exc.value)
        assert SECRET_GROQ not in str(exc.value)

    @pytest.mark.parametrize("provider_cls", [GeminiLLMClient, GroqLLMClient])
    def test_server_error_sanitized(self, monkeypatch, provider_cls) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        monkeypatch.setenv("GROQ_API_KEY", SECRET_GROQ)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text=f"boom api_key={SECRET_GEMINI}")

        provider = provider_cls(transport=httpx.MockTransport(handler))
        with pytest.raises(LLMClientError) as exc:
            provider.generate_structured(request=_request(), schema=LLMExtractionOutput)
        assert "server error" in str(exc.value).lower()
        assert SECRET_GEMINI not in str(exc.value)

    def test_gemini_model_not_found_message(self, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        provider = GeminiLLMClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
        with pytest.raises(LLMClientError) as exc:
            provider.generate_structured(request=_request(), schema=LLMExtractionOutput)
        assert "GEMINI_MODEL" in str(exc.value)

    def test_groq_model_not_found_message(self, monkeypatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", SECRET_GROQ)
        provider = GroqLLMClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
        with pytest.raises(LLMClientError) as exc:
            provider.generate_structured(request=_request(), schema=LLMExtractionOutput)
        assert "GROQ_MODEL" in str(exc.value)

    @pytest.mark.parametrize("provider_cls", [GeminiLLMClient, GroqLLMClient])
    def test_connection_error_sanitized(self, monkeypatch, provider_cls) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        monkeypatch.setenv("GROQ_API_KEY", SECRET_GROQ)

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom", request=request)

        provider = provider_cls(transport=httpx.MockTransport(handler))
        with pytest.raises(LLMClientError) as exc:
            provider.generate_structured(request=_request(), schema=LLMExtractionOutput)
        assert "could not reach" in str(exc.value).lower()
        assert SECRET_GEMINI not in str(exc.value)
        assert SECRET_GROQ not in str(exc.value)

    @pytest.mark.parametrize("provider_cls", [GeminiLLMClient, GroqLLMClient])
    def test_timeout_sanitized(self, monkeypatch, provider_cls) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        monkeypatch.setenv("GROQ_API_KEY", SECRET_GROQ)

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("timed out", request=request)

        provider = provider_cls(transport=httpx.MockTransport(handler))
        with pytest.raises(LLMClientError) as exc:
            provider.generate_structured(request=_request(), schema=LLMExtractionOutput)
        assert "did not respond in time" in str(exc.value).lower()
        assert SECRET_GEMINI not in str(exc.value)

    @pytest.mark.parametrize("provider_cls", [GeminiLLMClient, GroqLLMClient])
    def test_malformed_json_body(self, monkeypatch, provider_cls) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        monkeypatch.setenv("GROQ_API_KEY", SECRET_GROQ)
        provider = provider_cls(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, text="<html>not json</html>"))
        )
        with pytest.raises(LLMClientError) as exc:
            provider.generate_structured(request=_request(), schema=LLMExtractionOutput)
        assert "unreadable response" in str(exc.value)

    def test_gemini_unexpected_structure(self, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        provider = GeminiLLMClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"candidates": []}))
        )
        with pytest.raises(LLMClientError) as exc:
            provider.generate_structured(request=_request(), schema=LLMExtractionOutput)
        assert "no usable content" in str(exc.value)

    def test_groq_unexpected_structure(self, monkeypatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", SECRET_GROQ)
        provider = GroqLLMClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": []}))
        )
        with pytest.raises(LLMClientError) as exc:
            provider.generate_structured(request=_request(), schema=LLMExtractionOutput)
        assert "no usable content" in str(exc.value)

    def test_gemini_blocked_prompt(self, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        provider = GeminiLLMClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200,
                    json={"promptFeedback": {"blockReason": "SAFETY"}},
                )
            )
        )
        with pytest.raises(LLMClientError) as exc:
            provider.generate_structured(request=_request(), schema=LLMExtractionOutput)
        assert "blocked" in str(exc.value)

    def test_errors_are_controlled_hierarchy(self) -> None:
        from app.providers.llm import (
            LLMNotConfiguredError,
            LLMRequestError,
            LLMResponseError,
        )

        for cls in (LLMNotConfiguredError, LLMRequestError, LLMResponseError):
            assert issubclass(cls, LLMClientError)
        assert LLMClientError.code == "llm_provider_error"
        assert LLMNotConfiguredError.code == "llm_not_configured"
        assert LLMRequestError.code == "llm_request_failed"
        assert LLMResponseError.code == "llm_invalid_response"


# --------------------------------------------------------------------------
# Provider selection
# --------------------------------------------------------------------------


class TestProviderSelection:
    def test_default_is_stub(self, monkeypatch) -> None:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        assert config.llm_provider() == "stub"
        assert isinstance(build_llm_client(), StubLLMClient)

    def test_gemini_selected(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        assert config.llm_provider() == "gemini"
        assert isinstance(build_llm_client(), GeminiLLMClient)

    def test_groq_selected(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        assert isinstance(build_llm_client(), GroqLLMClient)

    def test_case_insensitive(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "Gemini")
        assert config.llm_provider() == "gemini"
        assert isinstance(build_llm_client(), GeminiLLMClient)

    def test_explicit_argument_overrides_env(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        assert isinstance(build_llm_client("stub"), StubLLMClient)
        assert isinstance(build_llm_client("groq"), GroqLLMClient)

    def test_unknown_provider_rejected(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "watson")
        with pytest.raises(ValueError):
            build_llm_client()

    def test_live_selection_needs_no_keys_at_build_time(self, monkeypatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        client = build_llm_client()
        assert isinstance(client, GeminiLLMClient)
        with pytest.raises(LLMProviderError) as exc:
            RequirementExtractionService(client).extract("Goa trip")
        assert exc.value.code == "llm_not_configured"


# --------------------------------------------------------------------------
# Extraction through mocked live providers
# --------------------------------------------------------------------------


class TestExtractionThroughLiveProviders:
    @pytest.mark.parametrize("provider", [_gemini_ok, _groq_ok])
    def test_valid_structured_extraction(self, monkeypatch, provider) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        monkeypatch.setenv("GROQ_API_KEY", SECRET_GROQ)
        service = RequirementExtractionService(provider())
        result = service.extract("Mumbai to Goa Dec 20-24 for 4 days, 2 people, around 30k")

        req = result.requirements
        assert req.trip.origin == "Mumbai"
        assert req.trip.destination == "Goa"
        assert req.trip.dates.date_range.start == date(2026, 12, 20)
        assert req.trip.dates.duration_days == 4
        assert req.travelers.count == 2
        assert req.budget.is_approximate is True
        assert req.hard_constraints.max_budget is None
        assert result.clarification_needed == []

    @pytest.mark.parametrize("provider", [_gemini_ok, _groq_ok])
    def test_hard_max_budget_sets_constraint(self, monkeypatch, provider) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        monkeypatch.setenv("GROQ_API_KEY", SECRET_GROQ)
        from decimal import Decimal

        client = provider(
            json.dumps(_json_output(budget={"amount": 30000, "currency": "INR", "kind": "hard_max"}))
        )
        req = RequirementExtractionService(client).extract("do not exceed 30k").requirements
        assert req.budget.is_approximate is False
        assert req.hard_constraints.max_budget == Decimal("30000")

    @pytest.mark.parametrize("provider", [_gemini_ok, _groq_ok])
    def test_missing_required_information_generates_clarifications(
        self, monkeypatch, provider
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        monkeypatch.setenv("GROQ_API_KEY", SECRET_GROQ)
        client = provider(json.dumps({"trip_destination": "Goa"}))
        result = RequirementExtractionService(client).extract("Goa trip")
        fields = {c.field for c in result.clarification_needed}
        assert "trip.origin" in fields
        assert "travel.timing" in fields
        assert result.requirements.trip.destination == "Goa"

    @pytest.mark.parametrize("provider", [_gemini_ok, _groq_ok])
    def test_empty_output_invents_nothing(self, monkeypatch, provider) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        monkeypatch.setenv("GROQ_API_KEY", SECRET_GROQ)
        client = provider("{}")
        result = RequirementExtractionService(client).extract("some vague request")
        req = result.requirements
        assert req.trip.origin is None
        assert req.trip.destination is None
        assert req.trip.dates is None
        assert req.travelers.count is None
        assert req.budget.total is None
        fields = {c.field for c in result.clarification_needed}
        assert {"trip.origin", "trip.destination", "travel.timing"} <= fields

    @pytest.mark.parametrize("provider", [_gemini_ok, _groq_ok])
    def test_ambiguous_request_preserves_clarification_items(self, monkeypatch, provider) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        monkeypatch.setenv("GROQ_API_KEY", SECRET_GROQ)
        payload = {
            "trip_origin": "Mumbai",
            "trip_destination": "Goa",
            "duration_days": 3,
            "traveler_count": 2,
            "clarification_needed": [
                {"field": "soft_preferences.comfort_level", "question": "Comfort level?"}
            ],
        }
        result = RequirementExtractionService(provider(json.dumps(payload))).extract("Mumbai to Goa")
        fields = [c.field for c in result.clarification_needed]
        assert "soft_preferences.comfort_level" in fields

    @pytest.mark.parametrize("provider", [_gemini_ok, _groq_ok])
    def test_malformed_llm_output_is_controlled_error(self, monkeypatch, provider) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        monkeypatch.setenv("GROQ_API_KEY", SECRET_GROQ)
        client = provider("this is not json at all")
        with pytest.raises(LLMProviderError) as exc:
            RequirementExtractionService(client).extract("Mumbai to Goa")
        assert exc.value.code == "llm_invalid_response"
        assert SECRET_GEMINI not in str(exc.value)
        assert SECRET_GROQ not in str(exc.value)

    @pytest.mark.parametrize("provider", [_gemini_ok, _groq_ok])
    def test_invalid_enum_is_rejected_by_validation(self, monkeypatch, provider) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        monkeypatch.setenv("GROQ_API_KEY", SECRET_GROQ)
        client = provider(json.dumps(_json_output(comfort_level="ultra_luxury")))
        with pytest.raises(LLMOutputValidationError):
            RequirementExtractionService(client).extract("Mumbai to Goa")

    @pytest.mark.parametrize("provider", [_gemini_ok, _groq_ok])
    def test_conflicting_dates_and_duration_rejected_deterministically(
        self, monkeypatch, provider
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        monkeypatch.setenv("GROQ_API_KEY", SECRET_GROQ)
        client = provider(json.dumps(_json_output(duration_days=10)))
        with pytest.raises(InconsistentExtractionError):
            RequirementExtractionService(client).extract("Mumbai to Goa Dec 20-24 for 10 days")

    def test_secret_never_leaks_through_service_errors(self, monkeypatch, caplog) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        failing = GeminiLLMClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(500, text=f"api_key={SECRET_GEMINI} trace")
            )
        )
        with pytest.raises(LLMProviderError) as exc:
            RequirementExtractionService(failing).extract("Mumbai to Goa")
        assert SECRET_GEMINI not in str(exc.value)
        assert SECRET_GEMINI not in caplog.text


# --------------------------------------------------------------------------
# Integration: agent + workflow with a mocked live provider
# --------------------------------------------------------------------------


class TestAgentIntegration:
    def test_missing_live_credentials_safely_fail_the_workflow(self, monkeypatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        extraction = RequirementExtractionService(GeminiLLMClient())
        agent = TravelAgent(extraction=extraction, store=InMemoryWorkflowStore())
        workflow_id = agent.create_workflow()
        response = agent.handle_message(workflow_id, "Mumbai to Goa")
        assert response.status is WorkflowStatus.FAILED
        assert response.error is not None
        assert response.error.code == "llm_not_configured"
        assert SECRET_GEMINI not in response.error.message
        assert response.search_handoff is None

    def test_live_provider_drives_complete_workflow(self, monkeypatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", SECRET_GEMINI)
        client = _gemini_ok()
        extraction = RequirementExtractionService(client)
        agent = TravelAgent(extraction=extraction, store=InMemoryWorkflowStore())
        workflow_id = agent.create_workflow()
        response = agent.handle_message(workflow_id, "Mumbai to Goa Dec 20-24, 4 days, 2 people")
        assert response.status is WorkflowStatus.REQUIREMENTS_COMPLETE
        assert response.action.value == "search"
        assert response.search_handoff is not None
        assert SECRET_GEMINI not in response.reply