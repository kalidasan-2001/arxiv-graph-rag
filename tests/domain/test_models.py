"""Validation and factory-construction tests for the domain models."""

import pytest
from pydantic import ValidationError

from app.domain.enums import (
    ConfidenceLevel,
    EntityType,
    EvidenceScoreKind,
    EvidenceType,
    RelationshipType,
    RetrievalStrategy,
    SectionType,
)
from app.domain.evidence import AnswerCitation, EvidenceItem, ResearchAnswer
from app.domain.ingestion import IngestionState
from app.domain.enums import IngestionStatus
from app.domain.knowledge import ScientificEntity, ScientificRelationship
from app.domain.papers import Paper, PaperChunk, PaperSection, PaperVersion
from app.domain.retrieval import RetrievalPlan


class TestPaperModel:
    def test_create_derives_stable_paper_id(self) -> None:
        paper = Paper.create(source="arxiv", source_id="2401.12345", title="A Paper")
        assert paper.paper_id == "paper:arxiv:2401.12345"

    def test_blank_title_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Paper.create(source="arxiv", source_id="2401.12345", title="   ")


class TestPaperVersionModel:
    def test_versions_of_same_paper_share_paper_id(self) -> None:
        paper = Paper.create(source="arxiv", source_id="2401.12345", title="A Paper")
        v1 = PaperVersion.create(paper_id=paper.paper_id, version="1")
        v2 = PaperVersion.create(paper_id=paper.paper_id, version="2")

        assert v1.paper_id == v2.paper_id == paper.paper_id
        assert v1.paper_version_id != v2.paper_version_id


class TestPaperSectionModel:
    def test_page_start_after_page_end_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PaperSection.create(
                paper_id="paper:arxiv:2401.12345",
                paper_version_id="paper-version:arxiv:2401.12345:v1",
                section_type=SectionType.METHODOLOGY,
                order=0,
                text="some text",
                page_start=10,
                page_end=5,
            )

    def test_blank_text_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PaperSection.create(
                paper_id="paper:arxiv:2401.12345",
                paper_version_id="paper-version:arxiv:2401.12345:v1",
                section_type=SectionType.METHODOLOGY,
                order=0,
                text="   ",
            )


class TestPaperChunkModel:
    def test_negative_chunk_index_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PaperChunk.create(
                paper_id="paper:arxiv:2401.12345",
                paper_version_id="paper-version:arxiv:2401.12345:v1",
                section_id="section:aaaa",
                section_type=SectionType.METHODOLOGY,
                chunk_index=-1,
                text="some text",
                token_count=10,
                chunk_config_fingerprint="fp-a",
            )

    def test_non_json_safe_metadata_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PaperChunk.create(
                paper_id="paper:arxiv:2401.12345",
                paper_version_id="paper-version:arxiv:2401.12345:v1",
                section_id="section:aaaa",
                section_type=SectionType.METHODOLOGY,
                chunk_index=0,
                text="some text",
                token_count=10,
                chunk_config_fingerprint="fp-a",
                metadata={"bad": object()},
            )

    def test_page_start_after_page_end_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PaperChunk.create(
                paper_id="paper:arxiv:2401.12345",
                paper_version_id="paper-version:arxiv:2401.12345:v1",
                section_id="section:aaaa",
                section_type=SectionType.METHODOLOGY,
                chunk_index=0,
                text="some text",
                token_count=10,
                chunk_config_fingerprint="fp-a",
                page_start=10,
                page_end=5,
            )


class TestScientificEntityModel:
    def test_blank_canonical_name_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScientificEntity.create(entity_type=EntityType.METHOD, canonical_name="   ")

    def test_duplicate_aliases_are_deduplicated(self) -> None:
        entity = ScientificEntity.create(
            entity_type=EntityType.METHOD,
            canonical_name="GraphRAG",
            aliases=["GraphRAG", "graphrag", "  GraphRAG  ", "Graph-based RAG"],
        )
        assert entity.aliases == ["GraphRAG", "Graph-based RAG"]


class TestScientificRelationshipModel:
    def test_confidence_below_zero_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScientificRelationship.create(
                source_entity_id="entity:paper:aaaa",
                relationship_type=RelationshipType.USES_METHOD,
                target_entity_id="entity:method:bbbb",
                confidence=-0.1,
            )

    def test_confidence_above_one_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScientificRelationship.create(
                source_entity_id="entity:paper:aaaa",
                relationship_type=RelationshipType.USES_METHOD,
                target_entity_id="entity:method:bbbb",
                confidence=1.1,
            )

    def test_boundary_confidence_values_are_accepted(self) -> None:
        low = ScientificRelationship.create(
            source_entity_id="entity:paper:aaaa",
            relationship_type=RelationshipType.USES_METHOD,
            target_entity_id="entity:method:bbbb",
            confidence=0.0,
        )
        high = ScientificRelationship.create(
            source_entity_id="entity:paper:aaaa",
            relationship_type=RelationshipType.USES_METHOD,
            target_entity_id="entity:method:bbbb",
            confidence=1.0,
        )
        assert low.confidence == 0.0
        assert high.confidence == 1.0


class TestIngestionStateModel:
    def test_negative_retry_count_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IngestionState(
                paper_id="paper:arxiv:2401.12345",
                paper_version_id="paper-version:arxiv:2401.12345:v1",
                status=IngestionStatus.DOWNLOADING,
                retry_count=-1,
            )

    def test_failed_status_requires_failed_stage(self) -> None:
        with pytest.raises(ValidationError):
            IngestionState(
                paper_id="paper:arxiv:2401.12345",
                paper_version_id="paper-version:arxiv:2401.12345:v1",
                status=IngestionStatus.FAILED,
            )

    def test_failed_status_with_stage_is_accepted(self) -> None:
        state = IngestionState(
            paper_id="paper:arxiv:2401.12345",
            paper_version_id="paper-version:arxiv:2401.12345:v1",
            status=IngestionStatus.FAILED,
            failed_stage=IngestionStatus.PARSING,
            failure_reason="malformed PDF",
        )
        assert state.failed_stage == IngestionStatus.PARSING


class TestRetrievalPlanModel:
    def test_non_positive_top_k_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalPlan(strategy=RetrievalStrategy.HYBRID, query="graph rag", top_k=0)

    def test_graph_depth_out_of_bounds_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalPlan(strategy=RetrievalStrategy.GRAPH, query="graph rag", graph_depth=0)
        with pytest.raises(ValidationError):
            RetrievalPlan(strategy=RetrievalStrategy.GRAPH, query="graph rag", graph_depth=99)

    def test_blank_query_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalPlan(strategy=RetrievalStrategy.VECTOR, query="   ")


class TestEvidenceItemModel:
    def test_requires_at_least_one_provenance_reference(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceItem.create(evidence_type=EvidenceType.TEXT, source="qdrant")

    def test_create_derives_deterministic_evidence_id(self) -> None:
        item = EvidenceItem.create(
            evidence_type=EvidenceType.TEXT, source="qdrant", chunk_id="chunk:aaaa"
        )
        same_item = EvidenceItem.create(
            evidence_type=EvidenceType.TEXT, source="qdrant", chunk_id="chunk:aaaa"
        )
        assert item.evidence_id == same_item.evidence_id

    def test_score_out_of_bounds_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceItem.create(
                evidence_type=EvidenceType.TEXT,
                source="qdrant",
                chunk_id="chunk:aaaa",
                score=1.5,
                score_kind=EvidenceScoreKind.VECTOR_SIMILARITY,
            )


class TestAnswerCitationModel:
    def test_blank_label_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AnswerCitation(
                citation_id="C1",
                evidence_id="evidence:aaaa",
                paper_id="paper:arxiv:2401.12345",
                label="   ",
            )


class TestResearchAnswerModel:
    def test_blank_answer_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResearchAnswer(
                answer="   ",
                confidence=ConfidenceLevel.HIGH,
                retrieval_strategy=RetrievalStrategy.HYBRID,
            )

    def test_insufficient_evidence_is_a_valid_confidence_level(self) -> None:
        answer = ResearchAnswer(
            answer="Insufficient evidence to answer this question.",
            confidence=ConfidenceLevel.INSUFFICIENT_EVIDENCE,
            retrieval_strategy=RetrievalStrategy.HYBRID,
        )
        assert answer.confidence == ConfidenceLevel.INSUFFICIENT_EVIDENCE
        assert answer.citations == []
