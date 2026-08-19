"""The LLM provider abstraction the rest of the application depends on.

Mirrors `app.embeddings.provider.EmbeddingProvider`: application code
depends only on this `Protocol`, never on a specific LLM SDK or HTTP client.
"""

from typing import Protocol, TypeVar

from pydantic import BaseModel

from app.core.config import get_settings
from app.core.exceptions import ConfigurationError

T = TypeVar("T", bound=BaseModel)


class LLMProvider(Protocol):
    """A configured LLM endpoint, ready to produce validated structured output."""

    @property
    def provider_name(self) -> str:
        """E.g. `"openai_compatible"` -- persisted as `extraction.llm_provider`."""
        ...

    @property
    def model_name(self) -> str:
        """E.g. `"gpt-4o-mini"` or a local LM Studio model id."""
        ...

    @property
    def provider_version(self) -> str | None:
        """Implementation version, where meaningful (prompt #22). `None`
        when genuinely not applicable (a raw HTTP client has no comparable
        "library version" the way `sentence-transformers` does) -- never
        fabricated."""
        ...

    @property
    def temperature(self) -> float:
        """The fixed sampling temperature every call uses -- exposed so
        callers can fold it into `extraction_config_fingerprint` (prompt
        #22) without needing raw `Settings` passed around separately."""
        ...

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        max_output_tokens: int | None = None,
    ) -> T:
        """Call the LLM and return output validated against `response_model`.

        Never returns unvalidated free-form text (prompt #7) -- malformed
        JSON or schema-validation failure is retried up to `LLM_MAX_RETRIES`
        times (prompt #38/#59), then raises `LLMResponseError`. Timeouts and
        transient transport failures are retried the same bounded number of
        times, then raise `LLMTimeoutError`/`LLMProviderError`.
        """
        ...


class LLMTokenUsage(BaseModel):
    """Optional per-call token usage, captured at the provider boundary
    when the backend reports it (added during live DeepSeek validation --
    DeepSeek's `/chat/completions` response includes a `usage` object;
    not every OpenAI-compatible server does). Deliberately **not** part of
    the `LLMProvider` Protocol itself: adding a required member would force
    every implementation (including `FakeLLMProvider`, used throughout the
    existing test suite) to supply one. A concrete provider may optionally
    expose a `.last_usage` property; callers that want it read it via
    `getattr(provider, "last_usage", None)` rather than assuming every
    provider reports it. This is raw token counts only -- no cost
    accounting is built on top of it."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def get_llm_provider() -> "LLMProvider":
    """FastAPI dependency (mirrors `get_embedding_provider`) -- no
    arguments, reads `Settings` itself, so tests can override it via
    `app.dependency_overrides` instead of a real LLM endpoint."""

    settings = get_settings()
    if settings.LLM_PROVIDER == "openai_compatible":
        # Imported here, not at module level, so importing this module
        # never requires `httpx` to already be configured/reachable.
        from app.llm.openai_compatible_provider import OpenAICompatibleLLMProvider

        return OpenAICompatibleLLMProvider(settings)

    raise ConfigurationError(f"unsupported LLM_PROVIDER: {settings.LLM_PROVIDER!r}")
