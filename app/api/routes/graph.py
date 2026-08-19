"""Read-only Neo4j knowledge-graph inspection routes.

Kept thin per CLAUDE.md #28: no Cypher here, no driver types -- everything
comes back through `GraphRepository`'s normalized DTOs (prompt #51/#54:
never raw driver records, never an arbitrary-Cypher endpoint).
"""

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.exceptions import GraphNotFoundError
from app.graph.models import GraphNodeRecord, GraphRelationshipRecord
from app.graph.neo4j_repository import get_graph_repository
from app.graph.repository import GraphRepository

router = APIRouter(prefix="/graph", tags=["graph"])


class GraphNodeResponse(BaseModel):
    entity_id: str
    entity_type: str
    canonical_name: str
    aliases: list[str]
    properties: dict[str, Any]


class GraphRelationshipResponse(BaseModel):
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence: float
    extraction_version: str | None
    source_chunk_id: str | None
    supporting_chunk_ids: list[str]
    provenance_type: str
    paper_version_id: str


def _node_response(node: GraphNodeRecord) -> GraphNodeResponse:
    return GraphNodeResponse(
        entity_id=node.entity_id,
        entity_type=node.entity_type,
        canonical_name=node.canonical_name,
        aliases=node.aliases,
        properties=node.properties,
    )


def _relationship_response(relationship: GraphRelationshipRecord) -> GraphRelationshipResponse:
    return GraphRelationshipResponse(
        relationship_id=relationship.relationship_id,
        source_entity_id=relationship.source_entity_id,
        target_entity_id=relationship.target_entity_id,
        relationship_type=relationship.relationship_type,
        confidence=relationship.confidence,
        extraction_version=relationship.extraction_version,
        source_chunk_id=relationship.source_chunk_id,
        supporting_chunk_ids=relationship.supporting_chunk_ids,
        provenance_type=relationship.provenance_type,
        paper_version_id=relationship.paper_version_id,
    )


class PaperGraphResponse(BaseModel):
    """Response body for `GET /api/v1/graph/papers/{paper_id}` (prompt #51):
    the paper's own node plus everything it's directly connected to --
    authors, cited papers, used methods, evaluated datasets, addressed
    tasks. Not a multi-hop traversal (that's Prompt 10's job)."""

    paper: GraphNodeResponse
    nodes: list[GraphNodeResponse]
    relationships: list[GraphRelationshipResponse]


@router.get("/papers/{paper_id}", response_model=PaperGraphResponse)
def get_paper_graph(
    paper_id: str, graph_repository: GraphRepository = Depends(get_graph_repository)
) -> PaperGraphResponse:
    """Inspect a paper's directly-connected knowledge graph (development/
    demo). Requires the paper to have already been graph-indexed
    (`POST /api/v1/papers/{paper_id}/graph-index`)."""

    paper, nodes, relationships = graph_repository.get_paper_graph(paper_id)
    if paper is None:
        raise GraphNotFoundError(f"paper {paper_id} has not been graph-indexed yet")

    return PaperGraphResponse(
        paper=_node_response(paper),
        nodes=[_node_response(node) for node in nodes],
        relationships=[_relationship_response(relationship) for relationship in relationships],
    )


class EntityGraphResponse(BaseModel):
    """Response body for `GET /api/v1/graph/entities/{entity_id}` (prompt
    #52): the entity itself, every node it's directly connected to (in
    either direction), and the relationships between them. This is what
    answers deterministic primitives like "which papers use Method X" or
    "which papers cite Paper X" (prompt #74) -- inbound relationships onto
    a Method/Dataset/Task/Paper entity."""

    entity: GraphNodeResponse
    connected_nodes: list[GraphNodeResponse]
    relationships: list[GraphRelationshipResponse]


@router.get("/entities/{entity_id}", response_model=EntityGraphResponse)
def get_entity_graph(
    entity_id: str, graph_repository: GraphRepository = Depends(get_graph_repository)
) -> EntityGraphResponse:
    """Inspect one canonical entity's directly-connected graph
    (development/demo)."""

    context = graph_repository.get_entity_context(entity_id)
    if context is None:
        raise GraphNotFoundError(f"entity {entity_id} not found in the graph")
    entity, connected_nodes, relationships = context

    return EntityGraphResponse(
        entity=_node_response(entity),
        connected_nodes=[_node_response(node) for node in connected_nodes],
        relationships=[_relationship_response(relationship) for relationship in relationships],
    )
