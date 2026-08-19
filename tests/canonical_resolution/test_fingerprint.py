"""Unit tests for canonicalization fingerprints (prompt #31/#32/#60)."""

from app.ingestion.canonical_resolution.fingerprint import (
    build_canonicalization_config_fingerprint,
    build_graph_index_generation_fingerprint,
)

_BASE_KWARGS = dict(
    canonicalization_version="v1",
    normalization_algorithm_version="nfkc-casefold-v1",
    alias_registry_version="v1",
    alias_registry_checksum="abc123",
    ontology_version="v1",
)


class TestCanonicalizationConfigFingerprint:
    def test_identical_inputs_produce_identical_fingerprints(self) -> None:
        assert build_canonicalization_config_fingerprint(
            **_BASE_KWARGS
        ) == build_canonicalization_config_fingerprint(**_BASE_KWARGS)

    def test_alias_registry_checksum_change_changes_the_fingerprint(self) -> None:
        """Prompt #60: changing the alias registry must change the
        fingerprint even if `alias_registry_version` itself doesn't move."""

        original = build_canonicalization_config_fingerprint(**_BASE_KWARGS)
        changed = build_canonicalization_config_fingerprint(
            **{**_BASE_KWARGS, "alias_registry_checksum": "def456"}
        )
        assert original != changed

    def test_canonicalization_version_change_changes_the_fingerprint(self) -> None:
        original = build_canonicalization_config_fingerprint(**_BASE_KWARGS)
        changed = build_canonicalization_config_fingerprint(
            **{**_BASE_KWARGS, "canonicalization_version": "v2"}
        )
        assert original != changed

    def test_normalization_algorithm_version_change_changes_the_fingerprint(self) -> None:
        original = build_canonicalization_config_fingerprint(**_BASE_KWARGS)
        changed = build_canonicalization_config_fingerprint(
            **{**_BASE_KWARGS, "normalization_algorithm_version": "nfkc-casefold-v2"}
        )
        assert original != changed

    def test_ontology_version_change_changes_the_fingerprint(self) -> None:
        original = build_canonicalization_config_fingerprint(**_BASE_KWARGS)
        changed = build_canonicalization_config_fingerprint(**{**_BASE_KWARGS, "ontology_version": "v2"})
        assert original != changed


class TestGraphIndexGenerationFingerprint:
    def test_identical_inputs_produce_identical_fingerprints(self) -> None:
        kwargs = dict(
            graph_extraction_artifact_checksum="checksum-a", canonicalization_config_fingerprint="config-a"
        )
        assert build_graph_index_generation_fingerprint(
            **kwargs
        ) == build_graph_index_generation_fingerprint(**kwargs)

    def test_different_artifact_checksum_changes_the_fingerprint(self) -> None:
        original = build_graph_index_generation_fingerprint(
            graph_extraction_artifact_checksum="checksum-a", canonicalization_config_fingerprint="config-a"
        )
        changed = build_graph_index_generation_fingerprint(
            graph_extraction_artifact_checksum="checksum-b", canonicalization_config_fingerprint="config-a"
        )
        assert original != changed

    def test_different_canonicalization_config_fingerprint_changes_the_fingerprint(self) -> None:
        original = build_graph_index_generation_fingerprint(
            graph_extraction_artifact_checksum="checksum-a", canonicalization_config_fingerprint="config-a"
        )
        changed = build_graph_index_generation_fingerprint(
            graph_extraction_artifact_checksum="checksum-a", canonicalization_config_fingerprint="config-b"
        )
        assert original != changed
