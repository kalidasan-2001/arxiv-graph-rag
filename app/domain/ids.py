"""Deterministic, storage-provider-independent domain identifiers.

These identifiers are the stable contract that PostgreSQL, Qdrant, and
Neo4j will later be reconciled against (CLAUDE.md #4, "Shared Identity
Model"). None of the ``build_*`` functions below depend on a database, an
embedding model, or an LLM -- they are pure functions of normalized domain
inputs.

ID scheme
---------
Two families of ids are used, chosen per identifier based on what actually
makes it stable:

* **Composed, human-readable ids** (``paper_id``, ``paper_version_id``) are
  built directly from an already-stable external identifier (e.g. an arXiv
  id), because reformatting a readable id into a hash adds no safety and
  only hurts debuggability. Example: ``paper:arxiv:2401.12345``,
  ``paper-version:arxiv:2401.12345:v2``.

* **Hash-derived ids** (``section_id``, ``chunk_id``, ``entity_id``,
  ``relationship_id``, ``evidence_id``, ``collection_id``) are built from a
  SHA-256 digest of their normalized identity inputs, truncated to 16 hex
  characters (64 bits -- ample collision resistance at the project's
  intended scale of 50-200 papers; see CLAUDE.md #44). Python's built-in
  ``hash()`` is never used for this because it is not stable across
  processes. Example: ``entity:method:9f2a1c7b3e4d5a60``.

``ingestion_job_id`` is the one exception to "deterministic": it identifies
an individual operational run rather than stable content, so it is
randomly generated -- retrying ingestion for the same paper must produce a
*different* job id, not the same one.
"""

import hashlib
import json
from typing import Any
from unicodedata import normalize as _unicode_normalize
from uuid import uuid4

from app.domain.enums import EntityType, EvidenceType, RelationshipType, SectionType

_HASH_LENGTH = 16  # hex chars => 64 bits of the underlying SHA-256 digest


def normalize_whitespace(text: str) -> str:
    """Trim and collapse internal whitespace without altering case.

    Used for display-safe fields (titles, names, labels) where identity
    should tolerate stray formatting but text should still read naturally.
    """

    return " ".join(text.split())


def normalize_identity_key(text: str) -> str:
    """Normalize text for identity/hashing purposes only.

    Unicode-normalizes (NFKC), collapses whitespace, and case-folds so that
    trivial formatting differences -- "GraphRAG" vs "GraphRAG " vs
    "graphrag" -- resolve to the same identity key. Must NOT be used for
    display text, only as input to hashing or equality comparison.
    """

    normalized = _unicode_normalize("NFKC", text)
    normalized = " ".join(normalized.split())
    return normalized.casefold()


def ensure_json_safe(value: dict[str, Any]) -> dict[str, Any]:
    """Validate that a metadata dict is plain-JSON-serializable.

    Shared by every domain model with a free-form ``metadata`` field so a
    vendor SDK object (e.g. a Qdrant point or Neo4j record) can never leak
    into a domain model undetected.
    """

    try:
        json.dumps(value)
    except TypeError as exc:
        raise ValueError(f"metadata must be JSON-serializable: {exc}") from exc
    return value


def _sha256_hex(*parts: str, length: int = _HASH_LENGTH) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:length]


def build_paper_id(source: str, source_id: str) -> str:
    """Build the logical paper identifier: ``paper:<source>:<source_id>``.

    All revisions of a paper (arXiv v1, v2, v3, ...) share this same id --
    it is the anchor that keeps "the paper" stable across re-indexing.
    """

    source_key = normalize_identity_key(source)
    source_id_key = normalize_whitespace(source_id)
    return f"paper:{source_key}:{source_id_key}"


def build_paper_version_id(paper_id: str, version: str) -> str:
    """Build a specific-revision identifier from a logical ``paper_id``.

    Example: ``build_paper_version_id("paper:arxiv:2401.12345", "2")``
    -> ``"paper-version:arxiv:2401.12345:v2"``.
    """

    version_key = normalize_whitespace(str(version))
    if not version_key.lower().startswith("v"):
        version_key = f"v{version_key}"
    _, _, suffix = paper_id.partition(":")
    return f"paper-version:{suffix}:{version_key}"


def build_section_id(paper_version_id: str, section_type: SectionType, order: int) -> str:
    """Build a section identifier from its position within a paper version."""

    digest = _sha256_hex(paper_version_id, section_type.value, str(order))
    return f"section:{digest}"


def build_chunk_id(
    paper_version_id: str, section_id: str, chunk_index: int, chunk_config_fingerprint: str
) -> str:
    """Build a chunk identifier.

    Deterministic from (paper version, section, chunk index, chunk config
    fingerprint) so re-running chunking with unchanged inputs *and*
    configuration reproduces the same ids instead of silently creating
    duplicate chunks (CLAUDE.md #13, idempotency). `chunk_config_fingerprint`
    is required, not optional: a bare `chunking_version` string is a
    human-maintained label nothing forces a developer to bump, so two
    genuinely different chunking configurations (chunk size, overlap,
    minimum size, tokenizer) sharing the same version could otherwise
    collide on identity despite holding completely different text.
    `build_chunk_config_fingerprint` (`app/ingestion/chunking/fingerprint.py`)
    derives this from every input that actually affects chunk output, so
    identity tracks the real configuration, not just its label. Passing the
    full fingerprint (rather than a pre-shortened one) is fine -- this
    function already truncates its own SHA-256 output to `_HASH_LENGTH`
    hex characters, which is the "stable shortened representation" the
    chunk_id itself needs.
    """

    digest = _sha256_hex(paper_version_id, section_id, str(chunk_index), chunk_config_fingerprint)
    return f"chunk:{digest}"


def build_entity_id(entity_type: EntityType, canonical_name: str) -> str:
    """Build a scientific-entity identifier.

    The entity type is included in the hash input (not just the id
    prefix), so e.g. ``Method: BERT`` and ``Dataset: BERT`` can never
    collide even if the string prefix format changes later.
    """

    name_key = normalize_identity_key(canonical_name)
    digest = _sha256_hex(entity_type.value, name_key)
    return f"entity:{entity_type.value}:{digest}"


def build_relationship_id(
    source_entity_id: str, relationship_type: RelationshipType, target_entity_id: str
) -> str:
    """Build a relationship identifier from (source, type, target)."""

    digest = _sha256_hex(source_entity_id, relationship_type.value, target_entity_id)
    return f"relationship:{digest}"


def build_evidence_id(evidence_type: EvidenceType, *reference_ids: str) -> str:
    """Build an evidence identifier from its type and the ids it references.

    Deterministic so evidence fusion (CLAUDE.md #23) can deduplicate the
    same underlying chunk/relationship surfaced by multiple retrieval
    passes, instead of treating repeated hits as distinct evidence.
    """

    digest = _sha256_hex(evidence_type.value, *reference_ids)
    return f"evidence:{digest}"


def build_collection_id(name: str) -> str:
    """Build a research-collection identifier from its normalized name."""

    digest = _sha256_hex(normalize_identity_key(name))
    return f"collection:{digest}"


def build_ingestion_job_id() -> str:
    """Build a new ingestion job identifier.

    Unlike the ids above, an ingestion job represents an individual
    operational run, not stable content -- retrying ingestion for the same
    paper must produce a distinct job id, so this is randomly generated
    rather than hashed from content.
    """

    return f"ingestion-job:{uuid4().hex}"
