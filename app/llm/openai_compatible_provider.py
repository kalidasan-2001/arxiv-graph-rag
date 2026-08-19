"""OpenAI-compatible chat-completions HTTP provider (prompt #43) -- V1's
only `LLMProvider` implementation. This is the sole module that speaks the
OpenAI HTTP API directly; nothing else in the codebase does (CLAUDE.md #15).

Talking plain OpenAI-compatible HTTP (via `httpx`, already a dependency)
rather than a vendor SDK is what lets this provider work unchanged against
LM Studio, a self-hosted OpenAI-compatible server, or OpenAI itself --
whichever `LLM_BASE_URL` points at (prompt #44). LM Studio is never
required; nothing here assumes it.
"""

import json
import time

import httpx
from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import (
    ConfigurationError,
    LLMProviderError,
    LLMResponseError,
    LLMTimeoutError,
)
from app.llm.provider import LLMTokenUsage, T

_PROVIDER_NAME = "openai_compatible"


class OpenAICompatibleLLMProvider:
    """Talks the `/chat/completions` OpenAI-compatible HTTP API."""

    def __init__(self, settings: Settings, *, client: httpx.Client | None = None) -> None:
        if not settings.LLM_BASE_URL:
            raise ConfigurationError(
                "LLM_BASE_URL is not configured (e.g. an LM Studio or "
                "OpenAI-compatible endpoint's /v1 URL)"
            )
        if not settings.LLM_MODEL:
            raise ConfigurationError("LLM_MODEL is not configured")

        self._base_url = settings.LLM_BASE_URL.rstrip("/")
        self._model = settings.LLM_MODEL
        self._api_key = settings.LLM_API_KEY
        self._temperature = settings.LLM_TEMPERATURE
        self._max_retries = settings.LLM_MAX_RETRIES
        # `client` is injectable (e.g. `httpx.Client(transport=httpx.MockTransport(...))`)
        # specifically so retry/parsing behavior is unit-testable without a
        # real endpoint (prompt #46) -- `get_llm_provider()` never passes this.
        self._client = client or httpx.Client(timeout=settings.LLM_TIMEOUT_SECONDS)
        self._last_usage: LLMTokenUsage | None = None

    @property
    def provider_name(self) -> str:
        return _PROVIDER_NAME

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_version(self) -> str | None:
        # A raw HTTP client talking a wire protocol has no comparable
        # "library version" the way `sentence-transformers` does -- the
        # protocol itself (OpenAI's chat-completions shape) is what would
        # need to change to affect output, and that's not a value this
        # provider can introspect. `None` here is documented absence, not
        # an oversight (prompt #22: provider_version only "where relevant").
        return None

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def last_usage(self) -> LLMTokenUsage | None:
        """Token usage from the most recent successful HTTP call, if the
        backend reported a `usage` object -- `None` otherwise (not every
        OpenAI-compatible server reports it). See `LLMTokenUsage`'s
        docstring for why this isn't part of the `LLMProvider` Protocol."""

        return self._last_usage

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        max_output_tokens: int | None = None,
    ) -> T:
        attempt = 0
        last_transport_error: Exception | None = None
        while attempt <= self._max_retries:
            attempt += 1
            try:
                raw_text = self._call_chat_completions(
                    system_prompt, user_prompt, max_output_tokens=max_output_tokens
                )
            except httpx.TimeoutException as exc:
                last_transport_error = exc
                if attempt <= self._max_retries:
                    continue
                raise LLMTimeoutError(
                    f"LLM request to {self._model!r} timed out after {attempt} attempt(s)"
                ) from exc
            except httpx.HTTPError as exc:
                last_transport_error = exc
                if attempt <= self._max_retries:
                    continue
                raise LLMProviderError(
                    f"LLM request to {self._model!r} failed after {attempt} attempt(s): {exc}"
                ) from exc

            try:
                parsed = _parse_structured_json(raw_text)
                return response_model.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                if attempt <= self._max_retries:
                    continue
                raise LLMResponseError(
                    f"LLM returned unparseable/invalid structured output after "
                    f"{attempt} attempt(s): {exc}"
                ) from exc

        # Unreachable in practice (every branch above either returns or
        # raises), but keeps the type checker honest about control flow.
        raise LLMProviderError(f"LLM request failed: {last_transport_error}")

    def _call_chat_completions(
        self, system_prompt: str, user_prompt: str, *, max_output_tokens: int | None = None
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "temperature": self._temperature,
            # Portable across OpenAI-compatible servers (including LM
            # Studio) -- plain JSON-object mode, not a provider-specific
            # strict function-calling/json-schema feature whose support
            # varies. Our own pydantic validation is the real enforcement.
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        response = self._client.post(
            f"{self._base_url}/chat/completions", json=payload, headers=headers
        )
        response.raise_for_status()
        data = response.json()

        # Captured whenever the backend reports it, regardless of whether
        # the content that follows turns out to parse/validate -- the
        # tokens were genuinely consumed by this HTTP call either way.
        usage = data.get("usage")
        if isinstance(usage, dict):
            self._last_usage = LLMTokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )

        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise LLMProviderError("LLM response message content was not a string")
        return content


def _parse_structured_json(raw_text: str) -> object:
    """Parse JSON with only harmless wrapper cleanup before schema validation."""

    text = _strip_single_json_fence(raw_text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        extracted = _extract_single_top_level_json_object(text)
        if extracted is None:
            raise exc
        return json.loads(extracted)


def _strip_single_json_fence(text: str) -> str:
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    opening = lines[0].strip().lower()
    closing = lines[-1].strip()
    if opening in {"```", "```json"} and closing == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _extract_single_top_level_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    end = None
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index
                break

    if end is None:
        return None

    prefix = text[:start].strip()
    suffix = text[end + 1 :].strip()
    wrapper = prefix + suffix
    if any(token in wrapper for token in "{}[]") or len(wrapper) > 160:
        return None

    candidate = text[start : end + 1].strip()
    trailing = text[end + 1 :]
    if "{" in trailing:
        return None
    return candidate
