"""The graph storage abstraction the rest of the application depends on.

Mirrors `app.storage.qdrant.repository.VectorRepository`: application code
depends only on this `Protocol`, never on `neo4j` driver types directly
(CLAUDE.md #15). Only the methods `GraphIndexingService` and the graph
inspection endpoints actually need -- not a universal graph ORM
(prompt #28).
"""

from typing import Protocol

from app.graph.models import (
    GraphNodeInput,
    GraphNodeRecord,
    GraphPathRecord,
    GraphRelationshipInput,
    GraphRelationshipRecord,
)


class GraphRepository(Protocol):
    """Storage operations for the canonical scientific knowledge graph."""

    def ensure_schema(self) -> None:
        """Idempotently create every uniqueness constraint/index this
        application needs (prompt #26/#27). Safe to call on every request --
        never drops/recreates existing schema."""
        ...

    def upsert_entities(self, nodes: list[GraphNodeInput]) -> None:
        """`MERGE` each node by its stable `entity_id` (never by
        `canonical_name` alone, prompt #39). Batched via `UNWIND`, not one
        transaction per node (prompt #38). Idempotent -- re-upserting the
        same nodes is always safe (prompt #48)."""
        ...

    def upsert_relationships(self, relationships: list[GraphRelationshipInput]) -> None:
        """`MERGE` each relationship by its stable `relationship_id`
        (prompt #25), never a Neo4j-internal relationship id. Endpoint
        nodes are matched, not created here -- callers must upsert
        entities first. Batched via `UNWIND` (prompt #38)."""
        ...

    def get_relationship_ids_for_generation(
        self, paper_version_id: str, *, generation_fingerprint: str
    ) -> set[str]:
        """The exact set of `relationship_id`s currently tagged with
        `generation_fingerprint` for `paper_version_id` -- the
        reconciliation/verification primitive (prompt #50): comparing this
        against the canonicalized graph's *expected* relationship ids is
        how VALID/MISSING/PARTIAL/STALE is distinguished, not a bare count
        (shared nodes make a global node count insufficient)."""
        ...

    def delete_generation(self, paper_version_id: str, *, exclude_generation_fingerprint: str) -> int:
        """Delete relationships tagged with `paper_version_id` whose
        generation fingerprint is *not* `exclude_generation_fingerprint`
        (prompt #46) -- removes a stale generation's paper-specific edges
        after its replacement has been verified complete. Never deletes
        nodes (prompt #47: orphan removal is optional and not implemented
        in V1 -- harmless orphans are preferred over risking a shared
        node). Always scoped to one `paper_version_id`; never touches
        another paper's edges to the same shared entity."""
        ...

    def get_entity(self, entity_id: str) -> GraphNodeRecord | None:
        """Read one node by its stable `entity_id`, or `None` if absent."""
        ...

    def get_paper_graph(
        self, paper_id: str
    ) -> tuple[GraphNodeRecord | None, list[GraphNodeRecord], list[GraphRelationshipRecord]]:
        """The paper's own node (or `None` if not indexed), every entity it
        is directly connected to, and every relationship between them --
        the normalized representation `GET /api/v1/graph/papers/{id}`
        returns (prompt #51). Never raw driver records."""
        ...

    def get_entity_context(
        self, entity_id: str
    ) -> tuple[GraphNodeRecord, list[GraphNodeRecord], list[GraphRelationshipRecord]] | None:
        """The entity itself, every directly connected node, and every
        relationship between them -- `None` if the entity doesn't exist.
        Backs `GET /api/v1/graph/entities/{id}` (prompt #52) and the graph
        query primitives (prompt #74: "which papers use Method X" is this
        entity's inbound `USES_METHOD` edges)."""
        ...

    def find_entities_by_canonical_name(
        self, canonical_name: str, *, entity_type: str | None = None, limit: int = 20
    ) -> list[GraphNodeRecord]:
        """Exact case-insensitive canonical-name lookup, optionally scoped
        by entity type. Used only after deterministic id lookup cannot be
        applied directly, so ambiguity can be surfaced instead of guessed."""
        ...

    def get_direct_paths(
        self,
        start_entity_id: str,
        *,
        relationship_type: str,
        direction: str,
        end_entity_type: str | None = None,
        limit: int = 20,
    ) -> list[GraphPathRecord]:
        """Return one-hop paths for one allowlisted relationship type.

        `direction` is either `outgoing` or `incoming`; relationship types
        remain closed over the ontology and are never caller-supplied
        Cypher.
        """
        ...

    def get_shared_entity_paths(
        self,
        paper_id: str,
        *,
        relationship_type: str,
        shared_entity_type: str,
        limit: int = 20,
    ) -> list[GraphPathRecord]:
        """Return `Paper -> shared entity <- other Paper` paths."""
        ...

    def get_citing_paper_entity_paths(
        self,
        paper_id: str,
        *,
        relationship_type: str,
        end_entity_type: str,
        limit: int = 20,
    ) -> list[GraphPathRecord]:
        """Return `Paper <- CITES <- citing Paper -> entity` paths."""
        ...

    def get_entity_paper_entity_paths(
        self,
        entity_id: str,
        *,
        incoming_relationship_type: str,
        outgoing_relationship_type: str,
        end_entity_type: str,
        limit: int = 20,
    ) -> list[GraphPathRecord]:
        """Return `Entity <- rel <- Paper -> rel -> Entity` paths."""
        ...

    def get_citation_neighborhood_paths(
        self, paper_id: str, *, depth: int = 1, limit: int = 20
    ) -> list[GraphPathRecord]:
        """Return bounded citation-only paths touching `paper_id`.

        V1 supports depth 1 and 2 only through explicitly bounded Cypher,
        never an unbounded arbitrary traversal.
        """
        ...
