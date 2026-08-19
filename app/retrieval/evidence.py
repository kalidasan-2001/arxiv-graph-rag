"""Unified evidence adapters and provenance bridge.

This module normalizes existing vector and graph retrieval outputs into the
shared `EvidenceItem` contract. It does not orchestrate retrieval across
stores and does not rank vector evidence against graph evidence.
"""

import logging
import time

from pydantic import BaseModel, Field

from app.core.exceptions import EvidenceGenerationMismatchError, EvidenceIdentityMismatchError
from app.domain.enums import EvidenceScoreKind, EvidenceSourceStore, EvidenceType
from app.domain.evidence import EvidenceItem, EvidenceProvenance
from app.domain.ids import build_evidence_id
from app.storage.qdrant.models import VectorChunkRecord, VectorSearchHit
from app.storage.qdrant.repository import VectorRepository

logger = logging.getLogger(__name__)


class SourceChunkReference(BaseModel):
    """Exact source chunk backing an evidence item."""

    chunk_id: str
    paper_id: str
    paper_version_id: str
    section_id: str
    section_type: str
    section_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    text: str
    vector_generation_fingerprint: str


class EvidenceBridgeResult(BaseModel):
    """Graph evidence plus any resolved source-text evidence."""

    graph_evidence: EvidenceItem
    source_chunks: list[SourceChunkReference] = Field(default_factory=list)
    text_evidence: list[EvidenceItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class VectorEvidenceAdapter:
    """Convert Qdrant hits into store-independent text evidence."""

    def from_hit(self, hit: VectorSearchHit) -> EvidenceItem:
        return text_evidence_from_hit(hit)


class GraphEvidenceAdapter:
    """Ensure graph evidence follows the unified evidence contract."""

    def from_graph_evidence(self, evidence: EvidenceItem) -> EvidenceItem:
        if evidence.evidence_type not in {
            EvidenceType.GRAPH_RELATIONSHIP,
            EvidenceType.GRAPH_PATH,
            EvidenceType.METADATA,
        }:
            raise ValueError(f"unsupported graph evidence type {evidence.evidence_type}")
        source_chunk_ids = _ordered_source_chunk_ids(evidence.metadata.get("relationships", []))
        paper_version_id = _first_relationship_value(evidence, "paper_version_id")
        provenance_type = _graph_provenance_type(evidence)
        path_confidence = float(evidence.metadata.get("path_confidence", 1.0))
        return evidence.model_copy(
            update={
                "paper_version_id": paper_version_id,
                "source_chunk_ids": source_chunk_ids,
                "score": path_confidence,
                "score_kind": EvidenceScoreKind.GRAPH_PATH_CONFIDENCE,
                "source_store": EvidenceSourceStore.NEO4J,
                "provenance": EvidenceProvenance(
                    provenance_type=provenance_type,
                    source_store=EvidenceSourceStore.NEO4J,
                    paper_id=evidence.paper_id,
                    paper_version_id=paper_version_id,
                    chunk_ids=source_chunk_ids,
                    relationship_ids=evidence.relationship_ids,
                    graph_index_generation_fingerprint=_first_relationship_value(
                        evidence, "graph_index_generation_fingerprint"
                    ),
                    extraction_version=_first_relationship_value(evidence, "extraction_version"),
                    provenance_complete=True,
                ),
                "metadata": {
                    **evidence.metadata,
                    "source_store": EvidenceSourceStore.NEO4J.value,
                    "score_kind": EvidenceScoreKind.GRAPH_PATH_CONFIDENCE.value,
                    "text_kind": evidence.metadata.get(
                        "text_kind", evidence.metadata.get("evidence_text_kind", "structural_summary")
                    ),
                    "provenance_complete": True,
                    "supporting_text_evidence_ids": list(evidence.supporting_text_evidence_ids),
                },
            }
        )


class EvidenceProvenanceBridge:
    """Resolve graph source chunk IDs to exact Qdrant text evidence."""

    def __init__(
        self,
        vector_repository: VectorRepository,
        *,
        max_supporting_chunks: int,
        expected_vector_generation_fingerprint: str | None = None,
    ) -> None:
        self._vector_repo = vector_repository
        self._max_supporting_chunks = max_supporting_chunks
        self._expected_vector_generation_fingerprint = expected_vector_generation_fingerprint
        self._graph_adapter = GraphEvidenceAdapter()

    def resolve_graph_evidence_sources(self, graph_evidence: EvidenceItem) -> EvidenceBridgeResult:
        started = time.monotonic()
        normalized_graph = self._graph_adapter.from_graph_evidence(graph_evidence)
        chunk_ids = _limit_chunk_ids(
            _ordered_chunk_ids(normalized_graph.source_chunk_ids), self._max_supporting_chunks
        )
        expected_versions = _expected_paper_version_by_chunk(normalized_graph)
        if not chunk_ids:
            warnings = []
            if normalized_graph.provenance and normalized_graph.provenance.provenance_type == "metadata":
                warnings.append("metadata evidence has no source chunk")
            bridged = self._with_bridge_metadata(normalized_graph, [], warnings)
            self._log(bridged, 0, 0, time.monotonic() - started)
            return EvidenceBridgeResult(graph_evidence=bridged, warnings=warnings)

        hits = self._vector_repo.get_by_chunk_ids(chunk_ids)
        by_chunk_id = {hit.chunk_id: hit for hit in hits}
        source_chunks: list[SourceChunkReference] = []
        text_evidence: list[EvidenceItem] = []
        warnings: list[str] = []

        for chunk_id in chunk_ids:
            hit = by_chunk_id.get(chunk_id)
            if hit is None:
                warnings.append(f"source chunk {chunk_id} was not found")
                continue
            self._validate_chunk(normalized_graph, hit, expected_versions)
            source_chunks.append(source_chunk_reference_from_chunk(hit))
            text_evidence.append(text_evidence_from_chunk(hit))

        bridged = self._with_bridge_metadata(
            normalized_graph,
            [item.evidence_id for item in text_evidence],
            warnings,
        )
        self._log(bridged, len(chunk_ids), len(source_chunks), time.monotonic() - started)
        return EvidenceBridgeResult(
            graph_evidence=bridged,
            source_chunks=source_chunks,
            text_evidence=text_evidence,
            warnings=warnings,
        )

    def _validate_chunk(
        self,
        graph_evidence: EvidenceItem,
        hit: VectorChunkRecord,
        expected_versions: dict[str, str],
    ) -> None:
        expected_paper_version_id = expected_versions.get(hit.chunk_id)
        if expected_paper_version_id is None and graph_evidence.provenance is not None:
            expected_paper_version_id = graph_evidence.provenance.paper_version_id
        if expected_paper_version_id is not None:
            if hit.paper_version_id != expected_paper_version_id:
                raise EvidenceIdentityMismatchError(
                    f"chunk {hit.chunk_id} belongs to paper_version_id={hit.paper_version_id}, "
                    f"expected {expected_paper_version_id}"
                )
        if self._expected_vector_generation_fingerprint is not None:
            if hit.vector_generation_fingerprint != self._expected_vector_generation_fingerprint:
                raise EvidenceGenerationMismatchError(
                    f"chunk {hit.chunk_id} belongs to vector generation "
                    f"{hit.vector_generation_fingerprint}, expected "
                    f"{self._expected_vector_generation_fingerprint}"
                )

    def _with_bridge_metadata(
        self, graph_evidence: EvidenceItem, supporting_text_ids: list[str], warnings: list[str]
    ) -> EvidenceItem:
        provenance_complete = not warnings
        provenance = graph_evidence.provenance
        if provenance is not None:
            provenance = provenance.model_copy(
                update={"provenance_complete": provenance_complete, "warnings": warnings}
            )
        return graph_evidence.model_copy(
            update={
                "supporting_text_evidence_ids": supporting_text_ids,
                "provenance": provenance,
                "metadata": {
                    **graph_evidence.metadata,
                    "supporting_text_evidence_ids": supporting_text_ids,
                    "provenance_complete": provenance_complete,
                    "provenance_warnings": warnings,
                },
            }
        )

    def _log(
        self, evidence: EvidenceItem, requested: int, resolved: int, duration_seconds: float
    ) -> None:
        logger.info(
            "evidence provenance bridge evidence_type=%s source_store=%s paper_id=%s "
            "source_chunks_requested=%d source_chunks_resolved=%d provenance_complete=%s "
            "duration_ms=%d status=ok",
            evidence.evidence_type.value,
            evidence.source_store.value if evidence.source_store else evidence.source,
            evidence.paper_id,
            requested,
            resolved,
            evidence.provenance.provenance_complete if evidence.provenance else False,
            int(duration_seconds * 1000),
        )


def text_evidence_from_hit(hit: VectorSearchHit) -> EvidenceItem:
    evidence_id = build_text_evidence_id(
        hit.chunk_id, hit.vector_generation_fingerprint
    )
    return EvidenceItem(
        evidence_id=evidence_id,
        evidence_type=EvidenceType.TEXT,
        paper_id=hit.paper_id,
        paper_version_id=hit.paper_version_id,
        chunk_id=hit.chunk_id,
        section_id=hit.section_id,
        section_type=hit.section_type,
        page_start=hit.page_start,
        page_end=hit.page_end,
        text=hit.text,
        score=hit.similarity_score,
        score_kind=EvidenceScoreKind.VECTOR_SIMILARITY,
        source="qdrant",
        source_store=EvidenceSourceStore.QDRANT,
        provenance=EvidenceProvenance(
            provenance_type="chunk",
            source_store=EvidenceSourceStore.QDRANT,
            paper_id=hit.paper_id,
            paper_version_id=hit.paper_version_id,
            chunk_ids=[hit.chunk_id],
            vector_generation_fingerprint=hit.vector_generation_fingerprint,
        ),
        metadata={
            "source_store": EvidenceSourceStore.QDRANT.value,
            "score_kind": EvidenceScoreKind.VECTOR_SIMILARITY.value,
            "section_id": hit.section_id,
            "section_type": hit.section_type,
            "section_title": hit.section_title,
            "chunk_index": hit.chunk_index,
            "page_start": hit.page_start,
            "page_end": hit.page_end,
            "vector_generation_fingerprint": hit.vector_generation_fingerprint,
            "text_kind": "source_chunk",
        },
    )


def text_evidence_from_chunk(chunk: VectorChunkRecord) -> EvidenceItem:
    evidence_id = build_text_evidence_id(
        chunk.chunk_id, chunk.vector_generation_fingerprint
    )
    return EvidenceItem(
        evidence_id=evidence_id,
        evidence_type=EvidenceType.TEXT,
        paper_id=chunk.paper_id,
        paper_version_id=chunk.paper_version_id,
        chunk_id=chunk.chunk_id,
        section_id=chunk.section_id,
        section_type=chunk.section_type,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        text=chunk.text,
        score=None,
        score_kind=None,
        source="qdrant",
        source_store=EvidenceSourceStore.QDRANT,
        provenance=EvidenceProvenance(
            provenance_type="chunk",
            source_store=EvidenceSourceStore.QDRANT,
            paper_id=chunk.paper_id,
            paper_version_id=chunk.paper_version_id,
            chunk_ids=[chunk.chunk_id],
            vector_generation_fingerprint=chunk.vector_generation_fingerprint,
        ),
        metadata={
            "source_store": EvidenceSourceStore.QDRANT.value,
            "score_kind": None,
            "section_id": chunk.section_id,
            "section_type": chunk.section_type,
            "section_title": chunk.section_title,
            "chunk_index": chunk.chunk_index,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "vector_generation_fingerprint": chunk.vector_generation_fingerprint,
            "text_kind": "source_chunk",
            "retrieval_context": "exact_chunk_lookup",
        },
    )


def source_chunk_reference_from_chunk(hit: VectorChunkRecord) -> SourceChunkReference:
    return SourceChunkReference(
        chunk_id=hit.chunk_id,
        paper_id=hit.paper_id,
        paper_version_id=hit.paper_version_id,
        section_id=hit.section_id,
        section_type=hit.section_type,
        section_title=hit.section_title,
        page_start=hit.page_start,
        page_end=hit.page_end,
        text=hit.text,
        vector_generation_fingerprint=hit.vector_generation_fingerprint,
    )


def build_text_evidence_id(chunk_id: str, vector_generation_fingerprint: str) -> str:
    return build_evidence_id(
        EvidenceType.TEXT,
        EvidenceSourceStore.QDRANT.value,
        chunk_id,
        vector_generation_fingerprint,
    )


def _ordered_source_chunk_ids(relationships: list[dict]) -> list[str]:
    ordered: list[str] = []
    for relationship in relationships:
        primary = relationship.get("source_chunk_id")
        supporting = sorted(relationship.get("supporting_chunk_ids") or [])
        for chunk_id in [primary, *supporting]:
            if chunk_id and chunk_id not in ordered:
                ordered.append(chunk_id)
    return ordered


def _ordered_chunk_ids(chunk_ids: list[str]) -> list[str]:
    ordered: list[str] = []
    for chunk_id in chunk_ids:
        if chunk_id not in ordered:
            ordered.append(chunk_id)
    return ordered


def _limit_chunk_ids(chunk_ids: list[str], limit: int) -> list[str]:
    return chunk_ids[:limit]


def _first_relationship_value(evidence: EvidenceItem, key: str) -> str | None:
    for relationship in evidence.metadata.get("relationships", []):
        value = relationship.get(key)
        if value:
            return str(value)
    return None


def _graph_provenance_type(evidence: EvidenceItem) -> str:
    types = {
        relationship.get("provenance_type")
        for relationship in evidence.metadata.get("relationships", [])
        if relationship.get("provenance_type")
    }
    if not types:
        return "graph"
    if len(types) == 1:
        return str(next(iter(types)))
    return "mixed"


def _expected_paper_version_by_chunk(evidence: EvidenceItem) -> dict[str, str]:
    expected: dict[str, str] = {}
    for relationship in evidence.metadata.get("relationships", []):
        paper_version_id = relationship.get("paper_version_id")
        if not paper_version_id:
            continue
        primary = relationship.get("source_chunk_id")
        supporting = relationship.get("supporting_chunk_ids") or []
        for chunk_id in [primary, *supporting]:
            if chunk_id:
                expected[str(chunk_id)] = str(paper_version_id)
    return expected
