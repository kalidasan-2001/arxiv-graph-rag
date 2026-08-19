"""Unit tests for `build_vector_generation_fingerprint` (prompt #25/#57)."""

from app.ingestion.vector_indexing.fingerprint import build_vector_generation_fingerprint


class TestDeterminism:
    def test_identical_inputs_produce_identical_fingerprint(self) -> None:
        first = build_vector_generation_fingerprint(
            chunk_artifact_checksum="aaa", embedding_config_fingerprint="bbb"
        )
        second = build_vector_generation_fingerprint(
            chunk_artifact_checksum="aaa", embedding_config_fingerprint="bbb"
        )
        assert first == second

    def test_fingerprint_is_a_sha256_hex_digest(self) -> None:
        fingerprint = build_vector_generation_fingerprint(
            chunk_artifact_checksum="aaa", embedding_config_fingerprint="bbb"
        )
        assert len(fingerprint) == 64
        assert all(c in "0123456789abcdef" for c in fingerprint)


class TestFieldSensitivity:
    def test_chunk_artifact_checksum_change_changes_fingerprint(self) -> None:
        first = build_vector_generation_fingerprint(
            chunk_artifact_checksum="aaa", embedding_config_fingerprint="bbb"
        )
        second = build_vector_generation_fingerprint(
            chunk_artifact_checksum="ccc", embedding_config_fingerprint="bbb"
        )
        assert first != second

    def test_embedding_config_fingerprint_change_changes_fingerprint(self) -> None:
        first = build_vector_generation_fingerprint(
            chunk_artifact_checksum="aaa", embedding_config_fingerprint="bbb"
        )
        second = build_vector_generation_fingerprint(
            chunk_artifact_checksum="aaa", embedding_config_fingerprint="ddd"
        )
        assert first != second

    def test_not_naive_concatenation(self) -> None:
        # "aaa"+"bbbccc" and "aaabbb"+"ccc" must not collide -- proves
        # canonical JSON serialization is actually used, not string glue.
        first = build_vector_generation_fingerprint(
            chunk_artifact_checksum="aaa", embedding_config_fingerprint="bbbccc"
        )
        second = build_vector_generation_fingerprint(
            chunk_artifact_checksum="aaabbb", embedding_config_fingerprint="ccc"
        )
        assert first != second
