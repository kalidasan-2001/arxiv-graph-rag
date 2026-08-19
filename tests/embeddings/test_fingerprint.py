"""Unit tests for `build_embedding_config_fingerprint` (prompt #57)."""

from app.embeddings.fingerprint import build_embedding_config_fingerprint


def _fingerprint(**overrides) -> str:
    defaults = dict(
        provider="sentence_transformers",
        model="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        normalize=True,
        provider_version="3.0.0",
    )
    defaults.update(overrides)
    return build_embedding_config_fingerprint(**defaults)


class TestDeterminism:
    def test_identical_inputs_produce_identical_fingerprint(self) -> None:
        assert _fingerprint() == _fingerprint()

    def test_fingerprint_is_a_sha256_hex_digest(self) -> None:
        fingerprint = _fingerprint()
        assert len(fingerprint) == 64
        assert all(c in "0123456789abcdef" for c in fingerprint)


class TestFieldSensitivity:
    def test_model_change_changes_fingerprint(self) -> None:
        assert _fingerprint() != _fingerprint(model="a-different-model")

    def test_dimension_change_changes_fingerprint(self) -> None:
        assert _fingerprint() != _fingerprint(dimension=768)

    def test_normalize_change_changes_fingerprint(self) -> None:
        assert _fingerprint(normalize=True) != _fingerprint(normalize=False)

    def test_provider_change_changes_fingerprint(self) -> None:
        assert _fingerprint() != _fingerprint(provider="a-different-provider")

    def test_provider_version_change_changes_fingerprint(self) -> None:
        assert _fingerprint(provider_version="1.0.0") != _fingerprint(provider_version="2.0.0")

    def test_none_vs_explicit_provider_version_are_distinguishable(self) -> None:
        assert _fingerprint(provider_version=None) != _fingerprint(provider_version="1.0.0")
