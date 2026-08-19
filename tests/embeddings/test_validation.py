"""Unit tests for `_validate_vectors` (prompt #23) -- the real production
validation logic used by `SentenceTransformerEmbeddingProvider`, exercised
directly as a pure function so no model download is required."""

import pytest

from app.core.exceptions import EmbeddingDimensionError, EmbeddingProviderError
from app.embeddings.sentence_transformers_provider import _validate_vectors


class TestVectorCountValidation:
    def test_matching_count_and_dimension_passes(self) -> None:
        _validate_vectors([[0.1, 0.2], [0.3, 0.4]], expected_count=2, dimension=2)  # must not raise

    def test_too_few_vectors_is_rejected(self) -> None:
        with pytest.raises(EmbeddingDimensionError):
            _validate_vectors([[0.1, 0.2]], expected_count=2, dimension=2)

    def test_too_many_vectors_is_rejected(self) -> None:
        with pytest.raises(EmbeddingDimensionError):
            _validate_vectors([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], expected_count=2, dimension=2)


class TestVectorShapeValidation:
    def test_wrong_vector_length_is_rejected(self) -> None:
        with pytest.raises(EmbeddingDimensionError):
            _validate_vectors([[0.1, 0.2, 0.3]], expected_count=1, dimension=2)


class TestMalformedValueValidation:
    def test_nan_is_rejected(self) -> None:
        with pytest.raises(EmbeddingProviderError):
            _validate_vectors([[0.1, float("nan")]], expected_count=1, dimension=2)

    def test_positive_infinity_is_rejected(self) -> None:
        with pytest.raises(EmbeddingProviderError):
            _validate_vectors([[0.1, float("inf")]], expected_count=1, dimension=2)

    def test_negative_infinity_is_rejected(self) -> None:
        with pytest.raises(EmbeddingProviderError):
            _validate_vectors([[float("-inf"), 0.1]], expected_count=1, dimension=2)
