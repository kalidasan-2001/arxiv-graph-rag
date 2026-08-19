"""Unit tests for `OpenAICompatibleLLMProvider` using `httpx.MockTransport`
-- no real network, no LM Studio, deterministic retry/parsing behavior.
"""

import json

import httpx
import pytest
from pydantic import BaseModel

from app.core.config import Settings
from app.core.exceptions import ConfigurationError, LLMProviderError, LLMResponseError, LLMTimeoutError
from app.llm.openai_compatible_provider import OpenAICompatibleLLMProvider
from app.retrieval.planning import StructuredQueryAnalysis


class _Echo(BaseModel):
    """A minimal response schema -- keeps these tests about provider
    mechanics, not about the graph-extraction schema specifically."""

    value: str


def _settings(**overrides) -> Settings:
    defaults = dict(
        LLM_BASE_URL="http://fake-llm.local/v1",
        LLM_MODEL="fake-model",
        LLM_API_KEY=None,
        LLM_MAX_RETRIES=2,
        LLM_TIMEOUT_SECONDS=5,
        LLM_TEMPERATURE=0.0,
    )
    defaults.update(overrides)
    # `_env_file=None` is required, not just the explicit `LLM_API_KEY=None`
    # default above: `Settings` (pydantic-settings) reads the real local
    # `.env` for *any* field this call doesn't explicitly override, so
    # without this a developer's real `LLM_API_KEY` (or any other secret)
    # silently leaks into these tests -- confirmed the hard way when a
    # real key ended up in a test failure's assertion output during this
    # session's live DeepSeek validation. Every test in this module must
    # go through this helper, never construct `Settings(...)` directly.
    return Settings(_env_file=None, **defaults)


def _client_with_handler(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _chat_completion_response(
    content: str, status_code: int = 200, *, usage: dict | None = None
) -> httpx.Response:
    body: dict = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(status_code, json=body)


class TestConfigurationValidation:
    def test_missing_base_url_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            OpenAICompatibleLLMProvider(_settings(LLM_BASE_URL=None))

    def test_missing_model_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            OpenAICompatibleLLMProvider(_settings(LLM_MODEL=None))


class TestSuccessfulCall:
    def test_valid_response_is_parsed_and_validated(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return _chat_completion_response(json.dumps({"value": "hello"}))

        provider = OpenAICompatibleLLMProvider(_settings(), client=_client_with_handler(handler))
        result = provider.generate_structured(
            system_prompt="sys", user_prompt="usr", response_model=_Echo
        )

        assert result.value == "hello"
        assert len(calls) == 1

    def test_request_includes_configured_model_and_temperature(self) -> None:
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return _chat_completion_response(json.dumps({"value": "x"}))

        provider = OpenAICompatibleLLMProvider(
            _settings(LLM_MODEL="my-model", LLM_TEMPERATURE=0.0), client=_client_with_handler(handler)
        )
        provider.generate_structured(system_prompt="sys", user_prompt="usr", response_model=_Echo)

        assert captured["payload"]["model"] == "my-model"
        assert captured["payload"]["temperature"] == 0.0
        assert captured["payload"]["messages"][0] == {"role": "system", "content": "sys"}
        assert captured["payload"]["messages"][1] == {"role": "user", "content": "usr"}

    def test_api_key_sent_as_bearer_token_when_configured(self) -> None:
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("authorization")
            return _chat_completion_response(json.dumps({"value": "x"}))

        provider = OpenAICompatibleLLMProvider(
            _settings(LLM_API_KEY="secret-key"), client=_client_with_handler(handler)
        )
        provider.generate_structured(system_prompt="sys", user_prompt="usr", response_model=_Echo)

        assert captured["auth"] == "Bearer secret-key"

    def test_no_authorization_header_when_no_api_key(self) -> None:
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("authorization")
            return _chat_completion_response(json.dumps({"value": "x"}))

        provider = OpenAICompatibleLLMProvider(_settings(), client=_client_with_handler(handler))
        provider.generate_structured(system_prompt="sys", user_prompt="usr", response_model=_Echo)

        assert captured["auth"] is None


class TestTokenUsageCapture:
    """Added during live DeepSeek validation: DeepSeek's response includes
    a `usage` object; this is captured at the provider boundary via
    `.last_usage`, without changing `generate_structured`'s return type or
    any domain/extraction contract."""

    def test_usage_is_captured_when_backend_reports_it(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _chat_completion_response(
                json.dumps({"value": "x"}),
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )

        provider = OpenAICompatibleLLMProvider(_settings(), client=_client_with_handler(handler))
        assert provider.last_usage is None  # nothing captured before any call

        provider.generate_structured(system_prompt="sys", user_prompt="usr", response_model=_Echo)

        assert provider.last_usage is not None
        assert provider.last_usage.prompt_tokens == 10
        assert provider.last_usage.completion_tokens == 5
        assert provider.last_usage.total_tokens == 15

    def test_usage_is_none_when_backend_does_not_report_it(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _chat_completion_response(json.dumps({"value": "x"}))  # no usage field

        provider = OpenAICompatibleLLMProvider(_settings(), client=_client_with_handler(handler))
        provider.generate_structured(system_prompt="sys", user_prompt="usr", response_model=_Echo)

        assert provider.last_usage is None

    def test_usage_reflects_the_most_recent_call(self) -> None:
        responses = iter(
            [
                {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return _chat_completion_response(json.dumps({"value": "x"}), usage=next(responses))

        provider = OpenAICompatibleLLMProvider(_settings(), client=_client_with_handler(handler))
        provider.generate_structured(system_prompt="sys", user_prompt="usr", response_model=_Echo)
        provider.generate_structured(system_prompt="sys", user_prompt="usr", response_model=_Echo)

        assert provider.last_usage.total_tokens == 28  # the second call's, not the first's


class TestMalformedOutputRetry:
    def test_malformed_json_is_retried_then_succeeds(self) -> None:
        responses = iter(["not json at all", json.dumps({"value": "ok"})])

        def handler(request: httpx.Request) -> httpx.Response:
            return _chat_completion_response(next(responses))

        provider = OpenAICompatibleLLMProvider(_settings(LLM_MAX_RETRIES=2), client=_client_with_handler(handler))
        result = provider.generate_structured(system_prompt="sys", user_prompt="usr", response_model=_Echo)

        assert result.value == "ok"

    def test_malformed_json_exhausting_retries_raises_llm_response_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _chat_completion_response("still not json")

        provider = OpenAICompatibleLLMProvider(_settings(LLM_MAX_RETRIES=2), client=_client_with_handler(handler))
        with pytest.raises(LLMResponseError):
            provider.generate_structured(system_prompt="sys", user_prompt="usr", response_model=_Echo)

    def test_schema_validation_failure_is_retried_then_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            # Valid JSON, but missing the required "value" field.
            return _chat_completion_response(json.dumps({"wrong_field": "x"}))

        provider = OpenAICompatibleLLMProvider(_settings(LLM_MAX_RETRIES=1), client=_client_with_handler(handler))
        with pytest.raises(LLMResponseError):
            provider.generate_structured(system_prompt="sys", user_prompt="usr", response_model=_Echo)

    def test_retry_count_is_bounded(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _chat_completion_response("not json")

        provider = OpenAICompatibleLLMProvider(_settings(LLM_MAX_RETRIES=2), client=_client_with_handler(handler))
        with pytest.raises(LLMResponseError):
            provider.generate_structured(system_prompt="sys", user_prompt="usr", response_model=_Echo)

        assert call_count == 3  # 1 initial attempt + 2 retries, never unbounded

    def test_json_code_fence_is_stripped_before_validation(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _chat_completion_response('```json\n{"value": "ok"}\n```')

        provider = OpenAICompatibleLLMProvider(_settings(), client=_client_with_handler(handler))
        result = provider.generate_structured(system_prompt="sys", user_prompt="usr", response_model=_Echo)

        assert result.value == "ok"

    def test_single_json_object_can_be_extracted_from_harmless_wrapper_text(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _chat_completion_response('Here is the JSON:\n{"value": "ok"}')

        provider = OpenAICompatibleLLMProvider(_settings(), client=_client_with_handler(handler))
        result = provider.generate_structured(system_prompt="sys", user_prompt="usr", response_model=_Echo)

        assert result.value == "ok"

    def test_observed_invalid_planner_enum_and_entity_shape_still_fail(self) -> None:
        observed_failure = {
            "query": "Which datasets does Paper A evaluate on?",
            "intent": "get_paper_evaluation_datasets",
            "structural_retrieval_required": True,
            "entities": {"paper": "Paper A"},
            "proposed_strategy": "graph",
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return _chat_completion_response(json.dumps(observed_failure))

        provider = OpenAICompatibleLLMProvider(_settings(LLM_MAX_RETRIES=0), client=_client_with_handler(handler))
        with pytest.raises(LLMResponseError) as exc_info:
            provider.generate_structured(
                system_prompt="sys",
                user_prompt="usr",
                response_model=StructuredQueryAnalysis,
            )

        assert "get_paper_evaluation_datasets" in str(exc_info.value)
        assert "entities" in str(exc_info.value)


class TestTransportFailureRetry:
    def test_timeout_is_retried_then_raises_llm_timeout_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("simulated timeout")

        provider = OpenAICompatibleLLMProvider(_settings(LLM_MAX_RETRIES=1), client=_client_with_handler(handler))
        with pytest.raises(LLMTimeoutError):
            provider.generate_structured(system_prompt="sys", user_prompt="usr", response_model=_Echo)

    def test_timeout_recovers_on_retry(self) -> None:
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise httpx.TimeoutException("simulated timeout")
            return _chat_completion_response(json.dumps({"value": "recovered"}))

        provider = OpenAICompatibleLLMProvider(_settings(LLM_MAX_RETRIES=2), client=_client_with_handler(handler))
        result = provider.generate_structured(system_prompt="sys", user_prompt="usr", response_model=_Echo)

        assert result.value == "recovered"

    def test_http_error_status_raises_llm_provider_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="internal server error")

        provider = OpenAICompatibleLLMProvider(_settings(LLM_MAX_RETRIES=0), client=_client_with_handler(handler))
        with pytest.raises(LLMProviderError):
            provider.generate_structured(system_prompt="sys", user_prompt="usr", response_model=_Echo)
