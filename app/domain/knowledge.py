"""Scientific knowledge-graph domain models.

These model what Neo4j will eventually store (CLAUDE.md #3), but contain
no Neo4j-specific types. Entity resolution (merging aliases across papers)
is explicitly out of scope here -- see CLAUDE.md #20; this stage only
defines the shape and identity rules a resolver will later operate on.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.domain.enums import EntityType, RelationshipType
from app.domain.ids import (
    build_entity_id,
    build_relationship_id,
    ensure_json_safe,
    normalize_identity_key,
    normalize_whitespace,
)


class ScientificEntity(BaseModel):
    """A canonical scientific concept (paper, author, method, dataset, task).

    ``aliases`` are surface forms known to refer to this entity (e.g.
    "GraphRAG" and "graph-based RAG"); low-confidence merges are the
    resolver's responsibility, not this model's.
    """

    entity_id: str
    entity_type: EntityType
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("canonical_name")
    @classmethod
    def _canonical_name_required(cls, value: str) -> str:
        normalized = normalize_whitespace(value)
        if not normalized:
            raise ValueError("canonical_name must not be blank")
        return normalized

    @field_validator("aliases")
    @classmethod
    def _unique_normalized_aliases(cls, values: list[str]) -> list[str]:
        """Deduplicate aliases by normalized identity, preserving first form seen."""

        seen: set[str] = set()
        unique: list[str] = []
        for alias in values:
            normalized = normalize_whitespace(alias)
            if not normalized:
                continue
            key = normalize_identity_key(normalized)
            if key in seen:
                continue
            seen.add(key)
            unique.append(normalized)
        return unique

    @field_validator("metadata")
    @classmethod
    def _metadata_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_json_safe(value)

    @classmethod
    def create(
        cls, *, entity_type: EntityType, canonical_name: str, **kwargs: Any
    ) -> "ScientificEntity":
        """Construct a ``ScientificEntity`` with its id derived via `build_entity_id`."""

        return cls(
            entity_id=build_entity_id(entity_type, canonical_name),
            entity_type=entity_type,
            canonical_name=canonical_name,
            **kwargs,
        )


class ScientificRelationship(BaseModel):
    """A provenance-carrying edge between two ``ScientificEntity`` records.

    ``source_chunk_id`` and ``confidence`` exist so the system can always
    answer "why does this relationship exist?" (CLAUDE.md #6) -- a
    relationship extracted purely from unsupported LLM output without this
    provenance is not something this model is meant to represent silently.
    """

    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: RelationshipType
    source_chunk_id: str | None = None
    confidence: float
    extraction_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def _confidence_in_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return value

    @field_validator("metadata")
    @classmethod
    def _metadata_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_json_safe(value)

    @classmethod
    def create(
        cls,
        *,
        source_entity_id: str,
        relationship_type: RelationshipType,
        target_entity_id: str,
        confidence: float,
        **kwargs: Any,
    ) -> "ScientificRelationship":
        """Construct a relationship with its id derived via `build_relationship_id`."""

        return cls(
            relationship_id=build_relationship_id(
                source_entity_id, relationship_type, target_entity_id
            ),
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relationship_type=relationship_type,
            confidence=confidence,
            **kwargs,
        )
