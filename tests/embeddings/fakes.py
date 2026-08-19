"""Deterministic fake `EmbeddingProvider` implementations for tests
(prompt #56) -- no model download, no randomness, fast and reproducible.
"""

import hashlib
import math
from collections.abc import Sequence

from app.embeddings.fingerprint import build_embedding_config_fingerprint


class FakeEmbeddingProvider:
    """`text -> deterministic fixed-dimension vector` via a stable hash.

    Good for exercising batch handling, dimension/NaN validation, config
    fingerprints, and call counts -- not for meaningful semantic ranking
    (see `BagOfWordsEmbeddingProvider` for that).
    """

    def __init__(
        self,
        *,
        dimension: int = 8,
        model_name: str = "fake-model",
        provider_name: str = "fake",
        normalize: bool = False,
        provider_version: str | None = "1.0.0",
    ) -> None:
        self._dimension = dimension
        self._model_name = model_name
        self._provider_name = provider_name
        self._normalize = normalize
        self._provider_version = provider_version
        self._config_fingerprint = build_embedding_config_fingerprint(
            provider=provider_name,
            model=model_name,
            dimension=dimension,
            normalize=normalize,
            provider_version=provider_version,
        )
        # Call-count/inspection state (prompt #60's idempotency tests
        # assert on these directly, mirroring `CountingChunker`/`CountingParser`
        # from Prompts 5/6).
        self.embed_documents_calls: list[list[str]] = []
        self.embed_query_calls: list[str] = []

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def normalize(self) -> bool:
        return self._normalize

    @property
    def provider_version(self) -> str | None:
        return self._provider_version

    @property
    def config_fingerprint(self) -> str:
        return self._config_fingerprint

    @property
    def call_count(self) -> int:
        """Total number of `embed_documents` batches issued -- the
        idempotency assertion prompt #60 asks for."""
        return len(self.embed_documents_calls)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.embed_documents_calls.append(list(texts))
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls.append(text)
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [digest[i % len(digest)] / 255.0 for i in range(self._dimension)]
        if self._normalize:
            norm = math.sqrt(sum(v * v for v in values)) or 1.0
            values = [v / norm for v in values]
        return values


class BagOfWordsEmbeddingProvider:
    """A deterministic fake whose vectors genuinely reflect word overlap
    (unlike `FakeEmbeddingProvider`'s hash-based vectors) -- used only to
    verify ranking behavior (prompt #66) without depending on a real model.
    Cosine similarity between two texts' vectors increases with shared
    vocabulary words, so a designed "matching" chunk can be asserted to
    rank above unrelated ones deterministically.
    """

    def __init__(self, vocabulary: Sequence[str]) -> None:
        self._vocab = list(dict.fromkeys(word.lower() for word in vocabulary))
        self._dimension = len(self._vocab)
        self._config_fingerprint = build_embedding_config_fingerprint(
            provider="fake-bow", model="bow-v1", dimension=self._dimension,
            normalize=True, provider_version=None,
        )

    @property
    def provider_name(self) -> str:
        return "fake-bow"

    @property
    def model_name(self) -> str:
        return "bow-v1"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def normalize(self) -> bool:
        return True

    @property
    def provider_version(self) -> str | None:
        return None

    @property
    def config_fingerprint(self) -> str:
        return self._config_fingerprint

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        words = {word.lower().strip(".,!?():;") for word in text.split()}
        vector = [1.0 if word in words else 0.0 for word in self._vocab]
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
