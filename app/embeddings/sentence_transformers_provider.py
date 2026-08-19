"""Local `sentence-transformers` embedding provider (prompt #4) -- V1's
only implementation. This is the sole module that imports
`sentence_transformers`; nothing else in the codebase does (CLAUDE.md #15).
"""

import math
from collections.abc import Sequence

from app.core.config import Settings
from app.core.exceptions import EmbeddingDimensionError, EmbeddingModelLoadError, EmbeddingProviderError
from app.embeddings.fingerprint import build_embedding_config_fingerprint

_PROVIDER_NAME = "sentence_transformers"


class SentenceTransformerEmbeddingProvider:
    """Wraps a local `sentence-transformers` model.

    Lazy-loading (prompt #43): the model is not constructed until first
    needed (`.dimension`, `.config_fingerprint`, or an embed call) -- FastAPI
    startup and `/health` never pay the model-load cost. Loaded at most
    once per process and reused (prompt #44) -- no uncontrolled global
    state, since the instance itself (owned by whatever constructed it) is
    the cache.
    """

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.EMBEDDING_MODEL
        self._device = settings.EMBEDDING_DEVICE
        self._normalize = settings.EMBEDDING_NORMALIZE
        self._model = None  # type: ignore[assignment]  # lazy: sentence_transformers.SentenceTransformer | None
        self._dimension: int | None = None
        self._config_fingerprint: str | None = None

    @property
    def provider_name(self) -> str:
        return _PROVIDER_NAME

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def normalize(self) -> bool:
        return self._normalize

    @property
    def provider_version(self) -> str | None:
        # The installed sentence-transformers package version -- a real
        # library upgrade can change model output even with an unchanged
        # model name/revision, so this feeds the config fingerprint
        # (prompt #7's "provider implementation version where relevant").
        import sentence_transformers

        return sentence_transformers.__version__

    @property
    def dimension(self) -> int:
        self._ensure_model()
        assert self._dimension is not None
        return self._dimension

    @property
    def config_fingerprint(self) -> str:
        # Requires `dimension`, which requires the model -- computing this
        # is itself an allowed lazy-load trigger (see class docstring).
        self._ensure_model()
        assert self._config_fingerprint is not None
        return self._config_fingerprint

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure_model()
        raw = model.encode(list(texts), normalize_embeddings=self._normalize, show_progress_bar=False)
        vectors = raw.tolist()
        _validate_vectors(vectors, expected_count=len(texts), dimension=self.dimension)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        raw = model.encode([text], normalize_embeddings=self._normalize, show_progress_bar=False)
        vectors = raw.tolist()
        _validate_vectors(vectors, expected_count=1, dimension=self.dimension)
        return vectors[0]

    def _ensure_model(self):  # noqa: ANN202 -- returns sentence_transformers.SentenceTransformer, imported lazily
        if self._model is not None:
            return self._model

        # Imported here, not at module level: importing `sentence_transformers`
        # pulls in torch, which is a meaningfully heavy import -- deferring
        # it until a real embedding operation keeps `import app.main` (and
        # thus FastAPI startup / `/health`) fast even when no embedding
        # work has happened yet.
        from sentence_transformers import SentenceTransformer

        try:
            model = SentenceTransformer(self._model_name, device=self._device)
        except Exception as exc:
            raise EmbeddingModelLoadError(
                f"failed to load embedding model {self._model_name!r} "
                f"(may require network access on first use to download "
                f"from Hugging Face): {exc}"
            ) from exc

        # `get_embedding_dimension` is the current method name; older
        # `sentence-transformers>=3.0` releases (this project's declared
        # minimum) only have the pre-rename `get_sentence_embedding_dimension`.
        if hasattr(model, "get_embedding_dimension"):
            dimension = model.get_embedding_dimension()
        else:
            dimension = model.get_sentence_embedding_dimension()
        if dimension is None:
            raise EmbeddingModelLoadError(
                f"model {self._model_name!r} did not report a sentence embedding dimension"
            )

        self._model = model
        self._dimension = dimension
        self._config_fingerprint = build_embedding_config_fingerprint(
            provider=self.provider_name,
            model=self._model_name,
            dimension=dimension,
            normalize=self._normalize,
            provider_version=self.provider_version,
        )
        return model


def _validate_vectors(vectors: list[list[float]], *, expected_count: int, dimension: int) -> None:
    """Prompt #23: every batch is validated before it's allowed to reach
    Qdrant -- wrong count, wrong length, or a non-finite value all fail
    loudly here rather than silently indexing bad vectors."""

    if len(vectors) != expected_count:
        raise EmbeddingDimensionError(
            f"embedding provider returned {len(vectors)} vectors for {expected_count} input texts"
        )
    for vector in vectors:
        if len(vector) != dimension:
            raise EmbeddingDimensionError(
                f"embedding provider returned a vector of length {len(vector)}, expected {dimension}"
            )
        if any(not math.isfinite(value) for value in vector):
            raise EmbeddingProviderError("embedding provider returned a NaN or infinite value")
