"""Unit tests for extraction fingerprints (prompt #53/#57)."""

from app.ingestion.graph_extraction.fingerprint import (
    build_extraction_config_fingerprint,
    build_graph_extraction_generation_fingerprint,
)


def _config_fingerprint(**overrides) -> str:
    defaults = dict(
        extraction_version="v1",
        llm_provider="openai_compatible",
        llm_model="gpt-4o-mini",
        llm_provider_version=None,
        prompt_version="v1",
        schema_version="v1",
        temperature=0.0,
    )
    defaults.update(overrides)
    return build_extraction_config_fingerprint(**defaults)


class TestExtractionConfigFingerprintDeterminism:
    def test_identical_inputs_produce_identical_fingerprint(self) -> None:
        assert _config_fingerprint() == _config_fingerprint()

    def test_fingerprint_is_a_sha256_hex_digest(self) -> None:
        fingerprint = _config_fingerprint()
        assert len(fingerprint) == 64
        assert all(c in "0123456789abcdef" for c in fingerprint)


class TestExtractionConfigFieldSensitivity:
    def test_extraction_version_change_changes_fingerprint(self) -> None:
        assert _config_fingerprint() != _config_fingerprint(extraction_version="v2")

    def test_llm_provider_change_changes_fingerprint(self) -> None:
        assert _config_fingerprint() != _config_fingerprint(llm_provider="another_provider")

    def test_llm_model_change_changes_fingerprint(self) -> None:
        assert _config_fingerprint() != _config_fingerprint(llm_model="a-different-model")

    def test_prompt_version_change_changes_fingerprint(self) -> None:
        assert _config_fingerprint() != _config_fingerprint(prompt_version="v2")

    def test_schema_version_change_changes_fingerprint(self) -> None:
        assert _config_fingerprint() != _config_fingerprint(schema_version="v2")

    def test_temperature_change_changes_fingerprint(self) -> None:
        assert _config_fingerprint(temperature=0.0) != _config_fingerprint(temperature=0.5)

    def test_provider_version_change_changes_fingerprint(self) -> None:
        assert _config_fingerprint(llm_provider_version="1.0") != _config_fingerprint(
            llm_provider_version="2.0"
        )


class TestGenerationFingerprint:
    def test_identical_inputs_produce_identical_fingerprint(self) -> None:
        first = build_graph_extraction_generation_fingerprint(
            chunk_artifact_checksum="aaa", extraction_config_fingerprint="bbb"
        )
        second = build_graph_extraction_generation_fingerprint(
            chunk_artifact_checksum="aaa", extraction_config_fingerprint="bbb"
        )
        assert first == second

    def test_chunk_checksum_change_changes_fingerprint(self) -> None:
        first = build_graph_extraction_generation_fingerprint(
            chunk_artifact_checksum="aaa", extraction_config_fingerprint="bbb"
        )
        second = build_graph_extraction_generation_fingerprint(
            chunk_artifact_checksum="ccc", extraction_config_fingerprint="bbb"
        )
        assert first != second

    def test_extraction_config_fingerprint_change_changes_fingerprint(self) -> None:
        first = build_graph_extraction_generation_fingerprint(
            chunk_artifact_checksum="aaa", extraction_config_fingerprint="bbb"
        )
        second = build_graph_extraction_generation_fingerprint(
            chunk_artifact_checksum="aaa", extraction_config_fingerprint="ddd"
        )
        assert first != second
