"""The embedding provider abstraction the rest of the application depends on.

Mirrors `app.ingestion.chunking.chunker.ScientificChunker`: application code
depends only on this `Protocol`, never on a specific embedding library or
model class.
"""

from collections.abc import Sequence
from typing import Protocol

from app.core.config import get_settings
from app.core.exceptions import ConfigurationError


class EmbeddingProvider(Protocol):
    """A configured embedding model, ready to embed document/query text."""

    @property
    def provider_name(self) -> str:
        """E.g. `"sentence_transformers"` -- persisted as `embedding_provider`."""
        ...

    @property
    def model_name(self) -> str:
        """E.g. `"sentence-transformers/all-MiniLM-L6-v2"` -- persisted as `embedding_model`."""
        ...

    @property
    def dimension(self) -> int:
        """Vector length this provider produces. Never guessed manually
        (prompt #13) -- implementations report this from the loaded model."""
        ...

    @property
    def normalize(self) -> bool:
        """Whether output vectors are L2-normalized (prompt #54)."""
        ...

    @property
    def provider_version(self) -> str | None:
        """Implementation version, where meaningful (prompt #7) -- e.g. the
        installed embedding library's version. `None` only if genuinely
        not applicable, never fabricated."""
        ...

    @property
    def config_fingerprint(self) -> str:
        """Deterministic fingerprint of the complete effective embedding
        configuration (prompt #7) -- see `build_embedding_config_fingerprint`.
        What `VectorIndexingService` actually compares to decide whether an
        existing vector generation is still valid; `model_name` alone is not
        sufficient (prompt #8: dimension/normalization/provider-implementation
        changes must also invalidate).

        May require constructing the underlying model to determine
        `dimension` (prompt #43 lazy-loading) -- computing this is itself
        allowed to trigger that one-time load.
        """
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of chunk texts. The caller controls batch size
        (`VectorIndexingService`, via `EMBEDDING_BATCH_SIZE`) -- this method
        embeds exactly what it's given, once. Raises `EmbeddingDimensionError`
        if the provider's own output fails validation (count/shape/NaN/Inf)."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed one query string. Kept separate from `embed_documents`
        (prompt #37) since some models use different query/document prompts
        internally -- never assume the two are always interchangeable."""
        ...


def get_embedding_provider() -> "EmbeddingProvider":
    """FastAPI dependency (mirrors `get_arxiv_client`/`get_pdf_download_client`
    -- no arguments, reads `Settings` itself) so tests can override it via
    `app.dependency_overrides` instead of a real model loading over the
    network.

    A fresh instance per call -- the provider itself lazy-loads and caches
    its own model on first use, so there's no expensive work here regardless.
    """

    settings = get_settings()
    if settings.EMBEDDING_PROVIDER == "sentence_transformers":
        # Imported here, not at module level, so importing this module
        # (and thus the `EmbeddingProvider` Protocol it defines) never
        # pulls in sentence-transformers/torch unless this branch runs.
        from app.embeddings.sentence_transformers_provider import (
            SentenceTransformerEmbeddingProvider,
        )

        return SentenceTransformerEmbeddingProvider(settings)

    raise ConfigurationError(f"unsupported EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER!r}")
