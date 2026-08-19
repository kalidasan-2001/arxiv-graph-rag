"""Integration coverage for exact Qdrant-backed evidence provenance bridging."""

from app.domain.enums import EvidenceType
from app.domain.evidence import EvidenceItem
from app.domain.ids import build_evidence_id
from app.retrieval.evidence import EvidenceProvenanceBridge, build_text_evidence_id
from app.storage.qdrant.models import VectorPoint, VectorPointPayload, build_qdrant_point_id
from app.storage.qdrant.qdrant_repository import QdrantVectorRepository


def _payload(chunk_id: str = "chunk:bridge") -> VectorPointPayload:
    return VectorPointPayload(
        chunk_id=chunk_id,
        paper_id="paper:arxiv:bridge",
        paper_version_id="paper-version:arxiv:bridge:v1",
        section_id="section:bridge-method",
        section_type="methodology",
        section_title="Method",
        chunk_index=0,
        page_start=5,
        page_end=6,
        source="arxiv",
        source_id="bridge",
        published_year=2024,
        categories=["cs.CL"],
        chunking_version="chunk-v1",
        chunk_config_fingerprint="chunk-fp",
        embedding_provider="fake",
        embedding_model="fake-model",
        embedding_config_fingerprint="embed-fp",
        vector_generation_fingerprint="vector-current",
        text="Bridge source text.",
    )


def _graph_evidence(chunk_id: str = "chunk:bridge") -> EvidenceItem:
    return EvidenceItem(
        evidence_id=build_evidence_id(
            EvidenceType.GRAPH_RELATIONSHIP,
            "paper_datasets",
            "paper:arxiv:bridge",
            "entity:dataset:bridge",
            "rel:bridge",
        ),
        evidence_type=EvidenceType.GRAPH_RELATIONSHIP,
        paper_id="paper:arxiv:bridge",
        chunk_id=chunk_id,
        entity_ids=["paper:arxiv:bridge", "entity:dataset:bridge"],
        relationship_ids=["rel:bridge"],
        text="Paper 'Bridge' evaluated on dataset 'Bridge Dataset'.",
        source="neo4j",
        metadata={
            "ordered_entity_ids": ["paper:arxiv:bridge", "entity:dataset:bridge"],
            "ordered_relationship_ids": ["rel:bridge"],
            "relationships": [
                {
                    "relationship_id": "rel:bridge",
                    "source_entity_id": "paper:arxiv:bridge",
                    "target_entity_id": "entity:dataset:bridge",
                    "relationship_type": "evaluated_on",
                    "source_chunk_id": chunk_id,
                    "supporting_chunk_ids": [chunk_id],
                    "confidence": 0.84,
                    "extraction_version": "extract-v1",
                    "paper_version_id": "paper-version:arxiv:bridge:v1",
                    "provenance_type": "chunk",
                    "graph_index_generation_fingerprint": "graph-current",
                }
            ],
            "path_confidence": 0.84,
            "evidence_text_kind": "structural_summary",
        },
    )


class TestQdrantBackedEvidenceBridge:
    def test_bridge_resolves_graph_chunk_to_text_evidence(
        self, qdrant_client, qdrant_collection_name
    ) -> None:
        repo = QdrantVectorRepository(qdrant_client, qdrant_collection_name)
        repo.ensure_collection(dimension=4, distance="cosine")
        payload = _payload()
        repo.upsert_chunks(
            [
                VectorPoint(
                    point_id=build_qdrant_point_id(payload.chunk_id),
                    vector=[1.0, 0.0, 0.0, 0.0],
                    payload=payload,
                )
            ]
        )

        result = EvidenceProvenanceBridge(
            repo,
            max_supporting_chunks=5,
            expected_vector_generation_fingerprint="vector-current",
        ).resolve_graph_evidence_sources(_graph_evidence())

        assert result.warnings == []
        assert result.source_chunks[0].chunk_id == "chunk:bridge"
        assert result.source_chunks[0].section_id == "section:bridge-method"
        assert result.source_chunks[0].page_start == 5
        assert result.text_evidence[0].evidence_id == build_text_evidence_id(
            "chunk:bridge", "vector-current"
        )
        assert result.graph_evidence.supporting_text_evidence_ids == [
            result.text_evidence[0].evidence_id
        ]
        assert result.graph_evidence.metadata["provenance_complete"] is True
