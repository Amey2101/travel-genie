from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from app.providers.llm import LLMExtractionOutput, LLMExtractionRequest

if TYPE_CHECKING:
    from app.providers.llm import T


class StubLLMClient:
    """Deterministic LLMClient that returns preconfigured structured output.

    This stub performs no semantic interpretation and makes no network calls.
    It is used exclusively for deterministic tests.
    """

    def __init__(
        self,
        output: LLMExtractionOutput | None = None,
        *,
        factory: Callable[[LLMExtractionRequest], LLMExtractionOutput] | None = None,
    ) -> None:
        self._output = output
        self._factory = factory
        self.last_request: LLMExtractionRequest | None = None

    def generate_structured(self, *, request: LLMExtractionRequest, schema: type[T]) -> T:
        self.last_request = request
        if self._factory is not None:
            out = self._factory(request)
        elif self._output is not None:
            out = self._output
        else:
            out = LLMExtractionOutput()
        return schema.model_validate(out)
