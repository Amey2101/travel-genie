"""Google Gemini implementation of the ``LLMClient`` boundary.

Uses the REST ``generateContent`` endpoint with a plain synchronous
``httpx.Client`` so it fits the existing synchronous ``LLMClient`` contract and
never runs ``asyncio.run`` inside a live event loop.

Safety properties (mirrors ``serpapi_discovery``):
- The API key is read server-side and sent only in the ``x-goog-api-key``
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

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


def _http_error(status: int) -> LLMRequestError:
    if status in (401, 403):
        return LLMRequestError(
            "The Gemini provider rejected the request "
            f"(HTTP {status}). Check GEMINI_API_KEY and model access."
        )
    if status == 404:
        return LLMRequestError(
            "The configured Gemini model was not found or is unavailable to "
            f"this key (HTTP {status}). Check GEMINI_MODEL."
        )
    if status == 429:
        return LLMRequestError(
            "The Gemini provider rate limit or quota was reached "
            f"(HTTP {status}). Please retry later."
        )
    if status >= 500:
        return LLMRequestError(
            f"The Gemini provider had a server error (HTTP {status}). "
            "Please retry later."
        )
    return LLMRequestError(
        f"The Gemini provider rejected the request (HTTP {status})."
    )


class GeminiLLMClient:
    """Structured-output ``LLMClient`` backed by Google Gemini."""

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
        return self._model or config.gemini_model()

    def _timeout_seconds(self) -> float:
        return (
            self._timeout
            if self._timeout is not None
            else config.gemini_timeout_seconds()
        )

    def generate_structured(
        self, *, request: LLMExtractionRequest, schema: type[T]
    ) -> T:
        key = config.gemini_key()
        if not key:
            raise LLMNotConfiguredError(
                "Live extraction is unavailable: no GEMINI_API_KEY is "
                "configured on the server. Set LLM_PROVIDER=stub for offline "
                "use or configure GEMINI_API_KEY."
            )

        model = self._model_name()
        url = GEMINI_API_URL.format(model=model)
        body = {
            "systemInstruction": {
                "parts": [{"text": structured_system_prompt(request.system_prompt, schema)}]
            },
            "contents": [
                {"role": "user", "parts": [{"text": request.user_message}]}
            ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": config.gemini_max_output_tokens(),
                "responseMimeType": "application/json",
            },
        }
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        response = self._post(url, body, headers)
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
                # httpx messages can embed request details; never surface them.
                raise LLMRequestError(
                    "The Gemini provider did not respond in time. Please retry."
                ) from exc
            except httpx.HTTPError as exc:
                raise LLMRequestError(
                    "Could not reach the Gemini provider. Please retry."
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
                "The Gemini provider returned an unreadable response."
            ) from exc
        if not isinstance(payload, dict):
            raise LLMResponseError(
                "The Gemini provider returned an unexpected response format."
            )

        feedback = payload.get("promptFeedback")
        if isinstance(feedback, dict) and feedback.get("blockReason"):
            raise LLMResponseError(
                "The Gemini provider blocked this request."
            )

        try:
            candidate = payload["candidates"][0]
            parts = candidate["content"]["parts"]
            text = "".join(
                part.get("text", "")
                for part in parts
                if isinstance(part, dict)
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                "The Gemini provider returned no usable content."
            ) from exc

        data = parse_json_object(text)
        return schema.model_validate(data)
