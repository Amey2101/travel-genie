"""Groq implementation of the ``LLMClient`` boundary.

Uses Groq's OpenAI-compatible ``/chat/completions`` REST API with a plain
synchronous ``httpx.Client`` so it fits the existing synchronous ``LLMClient``
contract.

Safety properties (mirrors ``serpapi_discovery``):
- The API key is read server-side and sent only in the ``Authorization``
  header; it is never logged, returned, or embedded in error messages.
- Provider/network errors are normalized to fixed, credential-free messages.
- Raw prompts and raw provider responses are never logged.
"""

from __future__ import annotations

import httpx

from app.core import config
from app.providers.llm import (
    LLMExtractionRequest,
    LLMNotConfiguredError,
    LLMRequestError,
    LLMResponseError,
    T,
    parse_json_object,
    structured_system_prompt,
)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _http_error(status: int) -> LLMRequestError:
    if status in (401, 403):
        return LLMRequestError(
            "The Groq provider rejected the request or denied access "
            f"(HTTP {status}). Check GROQ_API_KEY."
        )
    if status == 404:
        return LLMRequestError(
            "The configured Groq model was not found or is unavailable to "
            f"this key (HTTP {status}). Check GROQ_MODEL."
        )
    if status == 429:
        return LLMRequestError(
            "The Groq provider rate limit or quota was reached "
            f"(HTTP {status}). Please retry later."
        )
    if status >= 500:
        return LLMRequestError(
            f"The Groq provider had a server error (HTTP {status}). "
            "Please retry later."
        )
    return LLMRequestError(
        f"The Groq provider rejected the request (HTTP {status})."
    )


class GroqLLMClient:
    """Structured-output ``LLMClient`` backed by Groq (OpenAI-compatible)."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float | None = None,
        model: str | None = None,
    ) -> None:
        self._client = client
        self._transport = transport
        self._timeout = timeout
        self._model = model

    def _model_name(self) -> str:
        return self._model or config.groq_model()

    def _timeout_seconds(self) -> float:
        return (
            self._timeout
            if self._timeout is not None
            else config.groq_timeout_seconds()
        )

    def generate_structured(
        self, *, request: LLMExtractionRequest, schema: type[T]
    ) -> T:
        key = config.groq_key()
        if not key:
            raise LLMNotConfiguredError(
                "Live extraction is unavailable: no GROQ_API_KEY is "
                "configured on the server. Set LLM_PROVIDER=stub for offline "
                "use or configure GROQ_API_KEY."
            )

        body = {
            "model": self._model_name(),
            "messages": [
                {
                    "role": "system",
                    "content": structured_system_prompt(
                        request.system_prompt, schema
                    ),
                },
                {"role": "user", "content": request.user_message},
            ],
            "temperature": 0,
            "max_completion_tokens": config.groq_max_output_tokens(),
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        response = self._post(GROQ_API_URL, body, headers)
        return self._parse(response, schema)

    def _post(
        self, url: str, body: dict, headers: dict
    ) -> httpx.Response:
        own_client = self._client is None
        client = self._client or httpx.Client(
            transport=self._transport, timeout=self._timeout_seconds()
        )
        try:
            try:
                response = client.post(url, json=body, headers=headers)
            except httpx.TimeoutException as exc:
                raise LLMRequestError(
                    "The Groq provider did not respond in time. Please retry."
                ) from exc
            except httpx.HTTPError as exc:
                raise LLMRequestError(
                    "Could not reach the Groq provider. Please retry."
                ) from exc
        finally:
            if own_client:
                client.close()

        if response.status_code >= 400:
            raise _http_error(response.status_code)
        return response

    def _parse(self, response: httpx.Response, schema: type[T]) -> T:
        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMResponseError(
                "The Groq provider returned an unreadable response."
            ) from exc
        if not isinstance(payload, dict):
            raise LLMResponseError(
                "The Groq provider returned an unexpected response format."
            )
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                "The Groq provider returned no usable content."
            ) from exc
        if not isinstance(text, str):
            raise LLMResponseError(
                "The Groq provider returned no usable content."
            )
        data = parse_json_object(text)
        return schema.model_validate(data)
