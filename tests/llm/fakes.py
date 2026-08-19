"""Deterministic fake `LLMProvider` for tests (prompt #46) -- no live
cloud APIs, no LM Studio, no internet, no model downloads.
"""

from app.core.exceptions import LLMResponseError
from app.ingestion.graph_extraction.models import RawExtractionResponse
from app.llm.provider import T


class FakeLLMProvider:
    """Returns a pre-programmed `RawExtractionResponse` keyed by a
    substring of the chunk text embedded in the user prompt -- lets tests
    assert exactly what one chunk should extract to without any real
    model call. `responses_by_chunk_marker` keys are matched via
    substring containment against `user_prompt` (the chunk text is
    embedded there by `build_user_prompt`)."""

    def __init__(
        self,
        *,
        provider_name: str = "fake",
        model_name: str = "fake-model",
        provider_version: str | None = "1.0.0",
        temperature: float = 0.0,
        responses_by_chunk_marker: dict[str, RawExtractionResponse] | None = None,
        default_response: RawExtractionResponse | None = None,
        fail_on_call_numbers: set[int] | None = None,
    ) -> None:
        self._provider_name = provider_name
        self._model_name = model_name
        self._provider_version = provider_version
        self._temperature = temperature
        self._responses = responses_by_chunk_marker or {}
        self._default_response = default_response or RawExtractionResponse()
        self._fail_on_call_numbers = fail_on_call_numbers or set()
        self.calls: list[str] = []  # user_prompt per call, in order
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_version(self) -> str | None:
        return self._provider_version

    @property
    def temperature(self) -> float:
        return self._temperature

    def generate_structured(
        self, *, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> T:
        self.call_count += 1
        self.calls.append(user_prompt)
        if self.call_count in self._fail_on_call_numbers:
            raise LLMResponseError(
                f"fake provider configured to fail on call {self.call_count}"
            )

        for marker, response in self._responses.items():
            if marker in user_prompt:
                return response_model.model_validate(response.model_dump())
        return response_model.model_validate(self._default_response.model_dump())
