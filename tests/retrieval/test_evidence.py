import pytest

from app.core.exceptions import EvidenceGenerationMismatchError, EvidenceIdentityMismatchError
from app.domain.enums import EvidenceScoreKind, EvidenceSourceStore, EvidenceType
from app.domain.evidence import EvidenceItem, build_evidence_pool
from app.domain.ids import build_evidence_id
from app.retrieval.evidence import (
    EvidenceProvenanceBridge,
    GraphEvidenceAdapter,
    VectorEvidenceAdapter,
    build_text_evidence_id,
)
from app.storage.qdrant.models import VectorChunkRecord, VectorSearchHit


def _hit(
    chunk_id: str = "chunk:source-a",
    *,
    paper_id: str = "paper:arxiv:a",
    paper_version_id: str = "paper-version:arxiv:a:v1",
    generation: str = "vector-current",
    score: float = 0.72,
) -> VectorSearchHit:
    return VectorSearchHit(
        chunk_id=chunk_id,
        paper_id=paper_id,
        paper_version_id=paper_version_id,
        section_id="section:method",
        section_type="methodology",
        section_title="Method",
        chunk_index=0,
        page_start=3,
        page_end=4,
        text="The paper evaluates Smoke Method on Smoke Dataset.",
        vector_generation_fingerprint=generation,
        similarity_score=score,
    )


def _chunk(
    chunk_id: str = "chunk:source-a",
    *,
    paper_id: str = "paper:arxiv:a",
    paper_version_id: str = "paper-version:arxiv:a:v1",
    generation: str = "vector-current",
) -> VectorChunkRecord:
    return VectorChunkRecord(
        chunk_id=chunk_id,
        paper_id=paper_id,
        paper_version_id=paper_version_id,
        section_id="section:method",
        section_type="methodology",
        section_title="Method",
        chunk_index=0,
        page_start=3,
        page_end=4,
        text="The paper evaluates Smoke Method on Smoke Dataset.",
        vector_generation_fingerprint=generation,
    )


def _graph_evidence(
    *,
    source_chunk_id: str | None = "chunk:source-a",
    supporting_chunk_ids: list[str] | None = None,
    provenance_type: str = "chunk",
    paper_version_id: str = "paper-version:arxiv:a:v1",
) -> EvidenceItem:
    supporting_chunk_ids = supporting_chunk_ids if supporting_chunk_ids is not None else ["chunk:source-b", "chunk:source-a"]
    return EvidenceItem(
        evidence_id=build_evidence_id(
            EvidenceType.GRAPH_RELATIONSHIP,
            "paper_methods",
            "paper:arxiv:a",
            "entity:method:m1",
            "rel:a-method",
        ),
        evidence_type=EvidenceType.GRAPH_RELATIONSHIP,
        paper_id="paper:arxiv:a",
        chunk_id=source_chunk_id,
        entity_ids=["paper:arxiv:a", "entity:method:m1"],
        relationship_ids=["rel:a-method"],
        text="Paper 'A' uses method 'M1'.",
        source="neo4j",
        metadata={
            "ordered_entity_ids": ["paper:arxiv:a", "entity:method:m1"],
            "ordered_relationship_ids": ["rel:a-method"],
            "relationships": [
                {
                    "relationship_id": "rel:a-method",
                    "source_entity_id": "paper:arxiv:a",
                    "target_entity_id": "entity:method:m1",
                    "relationship_type": "uses_method",
                    "source_chunk_id": source_chunk_id,
                    "supporting_chunk_ids": supporting_chunk_ids,
                    "confidence": 0.87,
                    "extraction_version": "extract-v1",
                    "paper_version_id": paper_version_id,
                    "provenance_type": provenance_type,
                    "graph_index_generation_fingerprint": "graph-current",
                }
            ],
            "path_confidence": 0.87,
            "evidence_text_kind": "structural_summary",
        },
    )


class _FakeVectorRepository:
    def __init__(self, chunks: list[VectorChunkRecord]) -> None:
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.requested: list[str] = []

    def get_by_chunk_ids(self, chunk_ids: list[str]) -> list[VectorChunkRecord]:
        self.requested = chunk_ids
        return [self.chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunks]


class TestVectorEvidenceAdapter:
    def test_vector_conversion_preserves_source_identity_and_score_kind(self) -> None:
        hit = _hit()
        evidence = VectorEvidenceAdapter().from_hit(hit)

        assert evidence.evidence_type == EvidenceType.TEXT
        assert evidence.evidence_id == build_text_evidence_id("chunk:source-a", "vector-current")
        assert evidence.paper_id == hit.paper_id
        assert evidence.paper_version_id == hit.paper_version_id
        assert evidence.section_id == hit.section_id
        assert evidence.page_start == 3
        assert evidence.score == 0.72
        assert evidence.score_kind == EvidenceScoreKind.VECTOR_SIMILARITY
        assert evidence.source_store == EvidenceSourceStore.QDRANT
        assert evidence.provenance.vector_generation_fingerprint == "vector-current"


class TestGraphEvidenceAdapter:
    def test_graph_relationship_conversion_preserves_provenance(self) -> None:
        evidence = GraphEvidenceAdapter().from_graph_evidence(_graph_evidence())

        assert evidence.evidence_type == EvidenceType.GRAPH_RELATIONSHIP
        assert evidence.source_store == EvidenceSourceStore.NEO4J
        assert evidence.score == 0.87
        assert evidence.score_kind == EvidenceScoreKind.GRAPH_PATH_CONFIDENCE
        assert evidence.source_chunk_ids == ["chunk:source-a", "chunk:source-b"]
        assert evidence.provenance.graph_index_generation_fingerprint == "graph-current"
        assert evidence.provenance.extraction_version == "extract-v1"

    def test_metadata_provenance_does_not_require_chunk_support(self) -> None:
        evidence = GraphEvidenceAdapter().from_graph_evidence(
            _graph_evidence(source_chunk_id=None, supporting_chunk_ids=[], provenance_type="metadata")
        )

        assert evidence.source_chunk_ids == []
        assert evidence.provenance.provenance_type == "metadata"


class TestEvidenceProvenanceBridge:
    def test_exact_chunk_bridge_creates_same_text_evidence_id_and_support_link(self) -> None:
        repo = _FakeVectorRepository([_chunk("chunk:source-a"), _chunk("chunk:source-b")])
        bridge = EvidenceProvenanceBridge(
            repo, max_supporting_chunks=5, expected_vector_generation_fingerprint="vector-current"
        )

        result = bridge.resolve_graph_evidence_sources(_graph_evidence())

        assert repo.requested == ["chunk:source-a", "chunk:source-b"]
        assert [item.evidence_id for item in result.text_evidence] == [
            build_text_evidence_id("chunk:source-a", "vector-current"),
            build_text_evidence_id("chunk:source-b", "vector-current"),
        ]
        assert result.graph_evidence.supporting_text_evidence_ids == [
            build_text_evidence_id("chunk:source-a", "vector-current"),
            build_text_evidence_id("chunk:source-b", "vector-current"),
        ]
        assert result.graph_evidence.metadata["provenance_complete"] is True

    def test_missing_source_chunk_is_a_warning_not_dropped_graph_evidence(self) -> None:
        bridge = EvidenceProvenanceBridge(
            _FakeVectorRepository([]), max_supporting_chunks=5
        )

        result = bridge.resolve_graph_evidence_sources(_graph_evidence())

        assert result.graph_evidence.evidence_id
        assert result.text_evidence == []
        assert result.graph_evidence.metadata["provenance_complete"] is False
        assert result.warnings == [
            "source chunk chunk:source-a was not found",
            "source chunk chunk:source-b was not found",
        ]

    def test_generation_mismatch_is_fatal(self) -> None:
        bridge = EvidenceProvenanceBridge(
            _FakeVectorRepository([_chunk(generation="stale")]),
            max_supporting_chunks=5,
            expected_vector_generation_fingerprint="vector-current",
        )

        with pytest.raises(EvidenceGenerationMismatchError):
            bridge.resolve_graph_evidence_sources(
                _graph_evidence(source_chunk_id="chunk:source-a", supporting_chunk_ids=[])
            )

    def test_paper_version_mismatch_is_fatal(self) -> None:
        bridge = EvidenceProvenanceBridge(
            _FakeVectorRepository([_chunk(paper_version_id="paper-version:arxiv:b:v1")]),
            max_supporting_chunks=5,
        )

        with pytest.raises(EvidenceIdentityMismatchError):
            bridge.resolve_graph_evidence_sources(
                _graph_evidence(source_chunk_id="chunk:source-a", supporting_chunk_ids=[])
            )


class TestEvidencePool:
    def test_pool_deduplicates_and_assigns_deterministic_pool_ids(self) -> None:
        a = VectorEvidenceAdapter().from_hit(_hit("chunk:a"))
        b = VectorEvidenceAdapter().from_hit(_hit("chunk:b"))
        c = VectorEvidenceAdapter().from_hit(_hit("chunk:c"))

        pool = build_evidence_pool([a, b, a, c])
        same_pool = build_evidence_pool([a, b, a, c])

        assert [(item.pool_id, item.evidence.evidence_id) for item in pool.items] == [
            ("E1", a.evidence_id),
            ("E2", b.evidence_id),
            ("E3", c.evidence_id),
        ]
        assert [item.pool_id for item in pool.items] == [item.pool_id for item in same_pool.items]
