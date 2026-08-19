"""Unit tests for `build_chunk_config_fingerprint`. Pure function, no DB."""

from app.ingestion.chunking.fingerprint import build_chunk_config_fingerprint


def _fingerprint(**overrides) -> str:
    defaults = dict(
        chunking_version="v1",
        chunk_size_tokens=700,
        chunk_overlap_tokens=100,
        min_chunk_tokens=80,
        tokenizer_name="whitespace-v1",
        tokenizer_version=None,
    )
    defaults.update(overrides)
    return build_chunk_config_fingerprint(**defaults)


class TestDeterminism:
    def test_identical_inputs_produce_identical_fingerprint(self) -> None:
        assert _fingerprint() == _fingerprint()

    def test_stable_across_separate_calls_not_object_identity(self) -> None:
        # Two entirely independent calls -- nothing about Python object
        # identity (e.g. `hash()`) can leak into the result.
        first = build_chunk_config_fingerprint(
            chunking_version="v1", chunk_size_tokens=700, chunk_overlap_tokens=100,
            min_chunk_tokens=80, tokenizer_name="whitespace-v1", tokenizer_version=None,
        )
        second = build_chunk_config_fingerprint(
            chunking_version="v1", chunk_size_tokens=700, chunk_overlap_tokens=100,
            min_chunk_tokens=80, tokenizer_name="whitespace-v1", tokenizer_version=None,
        )
        assert first == second

    def test_fingerprint_is_a_sha256_hex_digest(self) -> None:
        fingerprint = _fingerprint()
        assert len(fingerprint) == 64
        assert all(c in "0123456789abcdef" for c in fingerprint)


class TestFieldSensitivity:
    def test_chunking_version_change_changes_fingerprint(self) -> None:
        assert _fingerprint() != _fingerprint(chunking_version="v2")

    def test_chunk_size_change_changes_fingerprint(self) -> None:
        assert _fingerprint() != _fingerprint(chunk_size_tokens=800)

    def test_overlap_change_changes_fingerprint(self) -> None:
        assert _fingerprint() != _fingerprint(chunk_overlap_tokens=150)

    def test_min_chunk_tokens_change_changes_fingerprint(self) -> None:
        assert _fingerprint() != _fingerprint(min_chunk_tokens=120)

    def test_tokenizer_name_change_changes_fingerprint(self) -> None:
        assert _fingerprint() != _fingerprint(tokenizer_name="alternate-tokenizer-v1")

    def test_tokenizer_version_change_changes_fingerprint(self) -> None:
        assert _fingerprint(tokenizer_version="1.0.0") != _fingerprint(tokenizer_version="2.0.0")

    def test_none_vs_explicit_tokenizer_version_are_distinguishable(self) -> None:
        assert _fingerprint(tokenizer_version=None) != _fingerprint(tokenizer_version="1.0.0")
