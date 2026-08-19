"""Maps a resolved `CanonicalGraph` (`app.domain.knowledge` models) onto
the flat `GraphNodeInput`/`GraphRelationshipInput` shape `GraphRepository`
actually writes (prompt #21-24).

This is the one place that decides exactly which metadata fields are
trusted enough to become a Neo4j property -- CLAUDE.md #3/#21: never full
paper/chunk text, never speculative fields, only what's explicitly listed
below.
"""

from app.domain.enums import EntityType, RelationshipType
from app.domain.knowledge import ScientificEntity, ScientificRelationship
from app.graph.models import GraphNodeInput, GraphRelationshipInput

# Only AUTHORED_BY is created purely from arXiv metadata (Prompt 8, no
# source chunk); every other relationship type carries chunk-level
# provenance (prompt #24).
_METADATA_PROVENANCE_RELATIONSHIP_TYPES = frozenset({RelationshipType.AUTHORED_BY})


def entity_to_node_input(entity: ScientificEntity) -> GraphNodeInput:
    """Build the Neo4j-ready node for one canonical entity. Non-`Paper`
    entities get no extra properties beyond identity/aliases (prompt #21:
    "limited useful metadata," not a generic metadata dump) -- Prompt 8's
    per-candidate `extraction_confidence`/`source_chunk_ids`/
    `evidence_quotes` metadata is deliberately left out of Neo4j: it's
    candidate-level extraction detail, already fully preserved in
    `graph_extraction.json`, not graph-level truth."""

    properties: dict = {}
    if entity.entity_type == EntityType.PAPER:
        if entity.metadata.get("source") is not None:
            properties["source"] = entity.metadata["source"]
        if entity.metadata.get("source_id") is not None:
            properties["source_id"] = entity.metadata["source_id"]
        if entity.metadata.get("published_at") is not None:
            properties["published_at"] = entity.metadata["published_at"]
        if entity.metadata.get("placeholder"):
            # A reference-only Paper node (prompt #19) -- never given a
            # `title`/author/abstract, since none of that is actually
            # known yet. A later real discovery of this same paper
            # (matching `paper_id`) enriches it in place via `MERGE`.
            properties["is_placeholder"] = True
        else:
            properties["title"] = entity.canonical_name
    return GraphNodeInput(
        entity_id=entity.entity_id,
        entity_type=entity.entity_type.value,
        canonical_name=entity.canonical_name,
        aliases=entity.aliases,
        properties=properties,
    )


def relationship_to_relationship_input(
    relationship: ScientificRelationship,
    *,
    paper_version_id: str,
    graph_index_generation_fingerprint: str,
) -> GraphRelationshipInput:
    provenance_type = (
        "metadata"
        if relationship.relationship_type in _METADATA_PROVENANCE_RELATIONSHIP_TYPES
        else "chunk"
    )
    return GraphRelationshipInput(
        relationship_id=relationship.relationship_id,
        source_entity_id=relationship.source_entity_id,
        target_entity_id=relationship.target_entity_id,
        relationship_type=relationship.relationship_type.value,
        confidence=relationship.confidence,
        extraction_version=relationship.extraction_version,
        source_chunk_id=relationship.source_chunk_id,
        supporting_chunk_ids=list(relationship.metadata.get("supporting_chunk_ids", [])),
        provenance_type=provenance_type,
        paper_version_id=paper_version_id,
        graph_index_generation_fingerprint=graph_index_generation_fingerprint,
    )
