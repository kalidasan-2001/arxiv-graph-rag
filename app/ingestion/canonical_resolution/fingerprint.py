"""Deterministic fingerprints for canonical resolution / graph indexing
(prompt #31/#32).

Mirrors `app.ingestion.graph_extraction.fingerprint` exactly:
`CANONICALIZATION_VERSION` alone is a human-maintained label nothing
forces a developer to bump when the normalization algorithm, alias
registry content, or ontology changes -- these fingerprints capture every
input that actually determines canonicalization output.
"""

import hashlib
import json

# Bumped only if `app.domain.ids.normalize_identity_key`'s actual
# algorithm changes (NFKC + whitespace-collapse + casefold) -- not on
# every unrelated commit.
NORMALIZATION_ALGORITHM_VERSION = "nfkc-casefold-v1"

# Bumped only if the ontology this resolver assumes (5 entity types, 5
# relationship types, the compatibility matrix) changes -- CLAUDE.md #5
# says not to expand it without a demonstrated need, so this should rarely move.
CANONICAL_ONTOLOGY_VERSION = "v1"


def build_canonicalization_config_fingerprint(
    *,
    canonicalization_version: str,
    normalization_algorithm_version: str,
    alias_registry_version: str,
    alias_registry_checksum: str,
    ontology_version: str,
) -> str:
    """SHA-256 hex digest of the canonical JSON of every field that
    materially affects canonicalization output. `alias_registry_checksum`
    (not just `alias_registry_version`) is included so adding/removing one
    alias entry changes this fingerprint even without a version bump
    (prompt #32/#60)."""

    canonical = {
        "canonicalization_version": canonicalization_version,
        "normalization_algorithm_version": normalization_algorithm_version,
        "alias_registry_version": alias_registry_version,
        "alias_registry_checksum": alias_registry_checksum,
        "ontology_version": ontology_version,
    }
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def build_graph_index_generation_fingerprint(
    *, graph_extraction_artifact_checksum: str, canonicalization_config_fingerprint: str
) -> str:
    """SHA-256 hex digest identifying "exactly which extraction artifact,
    canonicalized under exactly which resolver configuration" (prompt
    #31) -- mirrors
    `app.ingestion.graph_extraction.fingerprint.build_graph_extraction_generation_fingerprint`."""

    canonical = {
        "graph_extraction_artifact_checksum": graph_extraction_artifact_checksum,
        "canonicalization_config_fingerprint": canonicalization_config_fingerprint,
    }
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
