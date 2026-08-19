"""Deterministic Neo4j graph retrieval primitives (Prompt 10).

No LLM, no Qdrant, no hybrid ranking: callers choose an explicit graph
operation and this service turns bounded, allowlisted Neo4j paths into
provenance-preserving `EvidenceItem`s.
"""

import logging
import time
from enum import Enum

from pydantic import BaseModel, Field

from app.core.exceptions import GraphEntityAmbiguousError, GraphEntityNotFoundError, GraphSearchError
from app.domain.enums import EntityType, EvidenceType, RelationshipType
from app.domain.evidence import EvidenceItem
from app.domain.ids import build_evidence_id
from app.domain.knowledge import ScientificEntity
from app.graph.models import GraphNodeRecord, GraphPathRecord, GraphRelationshipRecord
from app.graph.repository import GraphRepository
from app.ingestion.canonical_resolution.alias_registry import (
    EntityAliasRegistry,
    get_default_alias_registry,
)
from app.ingestion.canonical_resolution.resolver import CanonicalEntityResolver
from app.retrieval.evidence import GraphEvidenceAdapter

logger = logging.getLogger(__name__)


class GraphSearchOperation(str, Enum):
    PAPER_METHODS = "paper_methods"
    PAPER_DATASETS = "paper_datasets"
    PAPER_TASKS = "paper_tasks"
    PAPER_AUTHORS = "paper_authors"
    PAPER_CITATIONS = "paper_citations"
    PAPER_CITED_BY = "paper_cited_by"
    PAPERS_FOR_METHOD = "papers_for_method"
    PAPERS_FOR_DATASET = "papers_for_dataset"
    PAPERS_FOR_TASK = "papers_for_task"
    SHARED_DATASETS = "shared_datasets"
    SHARED_METHODS = "shared_methods"
    DATASETS_FROM_CITING_PAPERS = "datasets_from_citing_papers"
    METHODS_FOR_DATASET = "methods_for_dataset"
    CITATION_NEIGHBORHOOD = "citation_neighborhood"


class GraphSearchResult(BaseModel):
    """One human-inspectable result backed by one evidence item."""

    entity: GraphNodeRecord | None = None
    path: GraphPathRecord
    evidence_id: str
    path_confidence: float
    summary: str


class GraphSearchResponseData(BaseModel):
    """Service result returned by API and scripts."""

    operation: GraphSearchOperation
    start_entity: GraphNodeRecord
    results: list[GraphSearchResult] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)


class GraphRetrievalService:
    """Application-level graph retrieval over explicit, bounded primitives."""

    def __init__(
        self,
        graph_repository: GraphRepository,
        *,
        max_depth: int,
        default_limit: int,
        max_limit: int,
        alias_registry: EntityAliasRegistry | None = None,
    ) -> None:
        self._graph_repo = graph_repository
        self._max_depth = max_depth
        self._default_limit = default_limit
        self._max_limit = max_limit
        self._resolver = CanonicalEntityResolver(alias_registry or get_default_alias_registry())
        self._evidence_adapter = GraphEvidenceAdapter()

    def search(
        self,
        *,
        operation: GraphSearchOperation,
        entity_id: str | None = None,
        entity_type: EntityType | None = None,
        canonical_name: str | None = None,
        depth: int | None = None,
        limit: int | None = None,
    ) -> GraphSearchResponseData:
        started = time.monotonic()
        resolved_depth = self._validate_depth(depth)
        resolved_limit = self._validate_limit(limit)
        start_entity = self._resolve_entity(
            entity_id=entity_id, entity_type=entity_type, canonical_name=canonical_name
        )

        paths = self._execute_operation(
            operation, start_entity=start_entity, depth=resolved_depth, limit=resolved_limit
        )
        evidence = [self._path_to_evidence(operation, path) for path in paths]
        evidence.sort(key=_evidence_sort_key)
        by_id = {item.evidence_id: item for item in evidence}
        results = [
            GraphSearchResult(
                entity=_result_entity_for_path(operation, path),
                path=path,
                evidence_id=item.evidence_id,
                path_confidence=item.metadata["path_confidence"],
                summary=item.text or "",
            )
            for path in paths
            for item in [by_id[self._path_evidence_id(operation, path)]]
        ]
        results.sort(key=lambda result: (len(result.path.relationships), -result.path_confidence, result.evidence_id))

        duration_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "graph retrieval operation=%s start_entity_id=%s depth=%d limit=%d "
            "results_count=%d paths_count=%d duration_ms=%d status=ok",
            operation.value,
            start_entity.entity_id,
            resolved_depth,
            resolved_limit,
            len(results),
            len(paths),
            duration_ms,
        )
        return GraphSearchResponseData(
            operation=operation, start_entity=start_entity, results=results, evidence=evidence
        )

    def _validate_depth(self, depth: int | None) -> int:
        resolved = 1 if depth is None else depth
        if resolved < 1 or resolved > self._max_depth:
            raise GraphSearchError(f"depth must be between 1 and {self._max_depth} (got {resolved})")
        return resolved

    def _validate_limit(self, limit: int | None) -> int:
        resolved = self._default_limit if limit is None else limit
        if resolved < 1 or resolved > self._max_limit:
            raise GraphSearchError(f"limit must be between 1 and {self._max_limit} (got {resolved})")
        return resolved

    def _resolve_entity(
        self,
        *,
        entity_id: str | None,
        entity_type: EntityType | None,
        canonical_name: str | None,
    ) -> GraphNodeRecord:
        if entity_id:
            entity = self._graph_repo.get_entity(entity_id)
            if entity is None:
                raise GraphEntityNotFoundError(f"graph entity {entity_id} was not found")
            return entity

        if not canonical_name:
            raise GraphSearchError("either entity_id or canonical_name must be provided")

        if entity_type is not None:
            candidate = ScientificEntity.create(entity_type=entity_type, canonical_name=canonical_name)
            resolved = self._resolver.resolve_entity(candidate).canonical_entity
            entity = self._graph_repo.get_entity(resolved.entity_id)
            if entity is not None:
                return entity

        candidates = self._graph_repo.find_entities_by_canonical_name(
            canonical_name, entity_type=entity_type.value if entity_type else None, limit=2
        )
        if not candidates:
            scope = f"{entity_type.value} " if entity_type else ""
            raise GraphEntityNotFoundError(f"graph entity {scope}{canonical_name!r} was not found")
        if len(candidates) > 1:
            raise GraphEntityAmbiguousError(
                f"graph entity name {canonical_name!r} is ambiguous",
                candidates=[
                    {
                        "entity_id": entity.entity_id,
                        "entity_type": entity.entity_type,
                        "canonical_name": entity.canonical_name,
                    }
                    for entity in candidates
                ],
            )
        return candidates[0]

    def _execute_operation(
        self,
        operation: GraphSearchOperation,
        *,
        start_entity: GraphNodeRecord,
        depth: int,
        limit: int,
    ) -> list[GraphPathRecord]:
        direct = {
            GraphSearchOperation.PAPER_METHODS: (RelationshipType.USES_METHOD, "outgoing", EntityType.METHOD),
            GraphSearchOperation.PAPER_DATASETS: (RelationshipType.EVALUATED_ON, "outgoing", EntityType.DATASET),
            GraphSearchOperation.PAPER_TASKS: (RelationshipType.ADDRESSES, "outgoing", EntityType.TASK),
            GraphSearchOperation.PAPER_AUTHORS: (RelationshipType.AUTHORED_BY, "outgoing", EntityType.AUTHOR),
            GraphSearchOperation.PAPER_CITATIONS: (RelationshipType.CITES, "outgoing", EntityType.PAPER),
            GraphSearchOperation.PAPER_CITED_BY: (RelationshipType.CITES, "incoming", EntityType.PAPER),
            GraphSearchOperation.PAPERS_FOR_METHOD: (RelationshipType.USES_METHOD, "incoming", EntityType.PAPER),
            GraphSearchOperation.PAPERS_FOR_DATASET: (RelationshipType.EVALUATED_ON, "incoming", EntityType.PAPER),
            GraphSearchOperation.PAPERS_FOR_TASK: (RelationshipType.ADDRESSES, "incoming", EntityType.PAPER),
        }
        if operation in direct:
            rel_type, direction, end_type = direct[operation]
            return self._graph_repo.get_direct_paths(
                start_entity.entity_id,
                relationship_type=rel_type.value,
                direction=direction,
                end_entity_type=end_type.value,
                limit=limit,
            )
        if operation == GraphSearchOperation.SHARED_DATASETS:
            return self._graph_repo.get_shared_entity_paths(
                start_entity.entity_id,
                relationship_type=RelationshipType.EVALUATED_ON.value,
                shared_entity_type=EntityType.DATASET.value,
                limit=limit,
            )
        if operation == GraphSearchOperation.SHARED_METHODS:
            return self._graph_repo.get_shared_entity_paths(
                start_entity.entity_id,
                relationship_type=RelationshipType.USES_METHOD.value,
                shared_entity_type=EntityType.METHOD.value,
                limit=limit,
            )
        if operation == GraphSearchOperation.DATASETS_FROM_CITING_PAPERS:
            return self._graph_repo.get_citing_paper_entity_paths(
                start_entity.entity_id,
                relationship_type=RelationshipType.EVALUATED_ON.value,
                end_entity_type=EntityType.DATASET.value,
                limit=limit,
            )
        if operation == GraphSearchOperation.METHODS_FOR_DATASET:
            return self._graph_repo.get_entity_paper_entity_paths(
                start_entity.entity_id,
                incoming_relationship_type=RelationshipType.EVALUATED_ON.value,
                outgoing_relationship_type=RelationshipType.USES_METHOD.value,
                end_entity_type=EntityType.METHOD.value,
                limit=limit,
            )
        if operation == GraphSearchOperation.CITATION_NEIGHBORHOOD:
            if depth > 2:
                raise GraphSearchError("citation_neighborhood supports depth 1 or 2 in V1")
            return self._graph_repo.get_citation_neighborhood_paths(
                start_entity.entity_id, depth=depth, limit=limit
            )
        raise GraphSearchError(f"unsupported graph search operation {operation.value}")

    def _path_to_evidence(self, operation: GraphSearchOperation, path: GraphPathRecord) -> EvidenceItem:
        entity_ids = [node.entity_id for node in path.nodes]
        relationship_ids = [relationship.relationship_id for relationship in path.relationships]
        source_chunk_ids = _source_chunk_ids(path.relationships)
        path_confidence = min((relationship.confidence for relationship in path.relationships), default=1.0)
        evidence_type = (
            EvidenceType.GRAPH_RELATIONSHIP
            if len(path.relationships) == 1
            else EvidenceType.GRAPH_PATH
        )
        evidence = EvidenceItem(
            evidence_id=self._path_evidence_id(operation, path),
            evidence_type=evidence_type,
            paper_id=_first_paper_id(path.nodes),
            chunk_id=source_chunk_ids[0] if len(source_chunk_ids) == 1 else None,
            entity_ids=entity_ids,
            relationship_ids=relationship_ids,
            text=_summarize_path(path),
            score=None,
            source="neo4j",
            metadata={
                "operation": operation.value,
                "nodes": [_node_metadata(node) for node in path.nodes],
                "relationships": [_relationship_metadata(relationship) for relationship in path.relationships],
                "ordered_entity_ids": entity_ids,
                "ordered_relationship_ids": relationship_ids,
                "source_chunk_ids": source_chunk_ids,
                "path_confidence": path_confidence,
                "evidence_text_kind": "structural_summary",
            },
        )
        return self._evidence_adapter.from_graph_evidence(evidence)

    def _path_evidence_id(self, operation: GraphSearchOperation, path: GraphPathRecord) -> str:
        return build_evidence_id(
            EvidenceType.GRAPH_RELATIONSHIP if len(path.relationships) == 1 else EvidenceType.GRAPH_PATH,
            operation.value,
            *[node.entity_id for node in path.nodes],
            *[relationship.relationship_id for relationship in path.relationships],
        )


def _source_chunk_ids(relationships: list[GraphRelationshipRecord]) -> list[str]:
    seen: set[str] = set()
    chunk_ids: list[str] = []
    for relationship in relationships:
        for chunk_id in [relationship.source_chunk_id, *relationship.supporting_chunk_ids]:
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                chunk_ids.append(chunk_id)
    return chunk_ids


def _first_paper_id(nodes: list[GraphNodeRecord]) -> str | None:
    return next((node.entity_id for node in nodes if node.entity_type == EntityType.PAPER.value), None)


def _node_metadata(node: GraphNodeRecord) -> dict:
    return {
        "entity_id": node.entity_id,
        "entity_type": node.entity_type,
        "canonical_name": node.canonical_name,
        "properties": node.properties,
    }


def _relationship_metadata(relationship: GraphRelationshipRecord) -> dict:
    return {
        "relationship_id": relationship.relationship_id,
        "source_entity_id": relationship.source_entity_id,
        "target_entity_id": relationship.target_entity_id,
        "relationship_type": relationship.relationship_type,
        "source_chunk_id": relationship.source_chunk_id,
        "supporting_chunk_ids": relationship.supporting_chunk_ids,
        "confidence": relationship.confidence,
        "extraction_version": relationship.extraction_version,
        "paper_version_id": relationship.paper_version_id,
        "provenance_type": relationship.provenance_type,
        "graph_index_generation_fingerprint": relationship.graph_index_generation_fingerprint,
    }


def _summarize_path(path: GraphPathRecord) -> str:
    if len(path.nodes) == 2 and len(path.relationships) == 1:
        source = _display_name(path.nodes[0])
        target = _display_name(path.nodes[1])
        relationship = path.relationships[0].relationship_type.replace("_", " ")
        return f"{path.nodes[0].entity_type.title()} '{source}' {relationship} {path.nodes[1].entity_type} '{target}'."
    pieces = [f"{node.entity_type}:{_display_name(node)}" for node in path.nodes]
    return "Graph path: " + " -> ".join(pieces) + "."


def _display_name(node: GraphNodeRecord) -> str:
    if node.entity_type == EntityType.PAPER.value:
        return str(node.properties.get("title") or node.properties.get("source_id") or node.canonical_name)
    return node.canonical_name


def _result_entity_for_path(operation: GraphSearchOperation, path: GraphPathRecord) -> GraphNodeRecord | None:
    if not path.nodes:
        return None
    if operation in {
        GraphSearchOperation.SHARED_DATASETS,
        GraphSearchOperation.SHARED_METHODS,
    }:
        return path.nodes[-1]
    return path.nodes[-1]


def _evidence_sort_key(item: EvidenceItem) -> tuple[int, float, str]:
    path_confidence = float(item.metadata.get("path_confidence", 0.0))
    return (len(item.relationship_ids), -path_confidence, item.evidence_id)
