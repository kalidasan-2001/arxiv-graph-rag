"""DTOs for the Neo4j graph store adapter.

`neo4j.graph.Node`/`Relationship` (and every other driver-specific class)
stop at `neo4j_repository.py` -- everything above that layer works with
these plain, provider-independent types instead (CLAUDE.md #15).

Node/relationship properties are deliberately *flat* (`str | int | float |
bool | None` or a `list` of one of those) -- Neo4j properties cannot be
nested maps, and CLAUDE.md #21/#3 already rules out storing large document
bodies here, so there is no format this stage needs richer than that.
"""

from typing import Any

from pydantic import BaseModel, Field

# What a Neo4j node/relationship property value is actually allowed to be.
GraphPropertyValue = str | int | float | bool | list[str] | None


class GraphNodeInput(BaseModel):
    """One canonical entity, ready to `MERGE` into Neo4j.

    `entity_type` selects the node label (`Paper`/`Author`/`Method`/
    `Dataset`/`Task`) -- Cypher labels can't be parameterized, so the
    repository groups nodes by this field before writing (prompt #21/#26).
    """

    entity_id: str
    entity_type: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    # Limited, trusted, flat metadata only (prompt #21) -- e.g. a Paper's
    # `source`/`source_id`/`title`/`published_at`. Never full text.
    properties: dict[str, GraphPropertyValue] = Field(default_factory=dict)


class GraphRelationshipInput(BaseModel):
    """One canonical, provenance-carrying relationship, ready to `MERGE`
    into Neo4j (prompt #22/#23/#25).

    `relationship_type` selects the Neo4j relationship type -- always one
    of the five ontology values, never a dynamic string from LLM output
    (prompt #22). `paper_version_id` scopes this edge to the paper version
    whose generation produced it -- the primitive stale-generation cleanup
    (prompt #44/#46) filters/deletes by.
    """

    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence: float
    extraction_version: str | None = None
    source_chunk_id: str | None = None
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    # "metadata" for the deterministic AUTHORED_BY edge (prompt #24),
    # "chunk" for every LLM/citation-derived semantic edge.
    provenance_type: str
    paper_version_id: str
    graph_index_generation_fingerprint: str


class GraphNodeRecord(BaseModel):
    """A node as read back from Neo4j (inspection/query paths) -- never a
    raw driver `Node` object crossing this boundary (prompt #51)."""

    entity_id: str
    entity_type: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphRelationshipRecord(BaseModel):
    """A relationship as read back from Neo4j (inspection/query paths)."""

    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence: float
    extraction_version: str | None = None
    source_chunk_id: str | None = None
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    provenance_type: str
    paper_version_id: str
    graph_index_generation_fingerprint: str


class GraphPathRecord(BaseModel):
    """A normalized, ordered graph path read from Neo4j.

    `nodes` are in traversal order and `relationships[i]` connects
    `nodes[i]` to `nodes[i + 1]` in that same conceptual path order. The
    relationship's stored source/target ids still preserve the true
    direction in Neo4j, which matters for reverse traversals such as
    `Paper <- CITES <- Paper`.
    """

    nodes: list[GraphNodeRecord]
    relationships: list[GraphRelationshipRecord]
