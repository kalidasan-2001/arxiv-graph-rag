"""Verify representative domain models serialize to plain JSON-compatible
dicts, without requiring a vendor-specific encoder (CLAUDE.md #17)."""

import json
from datetime import datetime, timezone

from app.domain.enums import (
    ConfidenceLevel,
    EntityType,
    EvidenceScoreKind,
    EvidenceType,
    RelationshipType,
    RetrievalStrategy,
    SectionType,
)
from app.domain.evidence import EvidenceItem, ResearchAnswer
from app.domain.knowledge import ScientificEntity, ScientificRelationship
from app.domain.papers import Paper, PaperChunk
from app.domain.retrieval import RetrievalPlan


def _round_trips_through_plain_json(model) -> dict:
    """Serialize via Pydantic's JSON mode and confirm `json.dumps` accepts
    the result untouched -- i.e. no vendor object survived serialization."""

    payload = model.model_dump(mode="json")
    json.dumps(payload)  # raises TypeError if anything isn't plain JSON
    return payload


class TestSerialization:
    def test_paper_serializes_cleanly(self) -> None:
        paper = Paper.create(
            source="arxiv",
            source_id="2401.12345",
            title="A Paper",
            authors=["Ada Lovelace"],
            published_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        payload = _round_trips_through_plain_json(paper)
        assert payload["paper_id"] == "paper:arxiv:2401.12345"
        assert payload["published_at"].startswith("2024-01-15")

    def test_paper_chunk_serializes_cleanly(self) -> None:
        chunk = PaperChunk.create(
            paper_id="paper:arxiv:2401.12345",
            paper_version_id="paper-version:arxiv:2401.12345:v1",
            section_id="section:aaaa",
            section_type=SectionType.METHODOLOGY,
            chunk_index=0,
            text="Some chunk text.",
            token_count=4,
            chunk_config_fingerprint="fp-a",
            metadata={"parser_version": "1"},
        )
        payload = _round_trips_through_plain_json(chunk)
        assert payload["section_type"] == "methodology"

    def test_scientific_entity_serializes_cleanly(self) -> None:
        entity = ScientificEntity.create(
            entity_type=EntityType.METHOD, canonical_name="GraphRAG"
        )
        payload = _round_trips_through_plain_json(entity)
        assert payload["entity_type"] == "method"

    def test_scientific_relationship_serializes_cleanly(self) -> None:
        relationship = ScientificRelationship.create(
            source_entity_id="entity:paper:aaaa",
            relationship_type=RelationshipType.USES_METHOD,
            target_entity_id="entity:method:bbbb",
            confidence=0.9,
        )
        payload = _round_trips_through_plain_json(relationship)
        assert payload["relationship_type"] == "uses_method"

    def test_retrieval_plan_serializes_cleanly(self) -> None:
        plan = RetrievalPlan(
            strategy=RetrievalStrategy.HYBRID, query="graph rag evaluation", top_k=5
        )
        payload = _round_trips_through_plain_json(plan)
        assert payload["strategy"] == "hybrid"

    def test_evidence_item_serializes_cleanly(self) -> None:
        evidence = EvidenceItem.create(
            evidence_type=EvidenceType.TEXT,
            source="qdrant",
            chunk_id="chunk:aaaa",
            score=0.8,
            score_kind=EvidenceScoreKind.VECTOR_SIMILARITY,
        )
        payload = _round_trips_through_plain_json(evidence)
        assert payload["evidence_type"] == "text"

    def test_research_answer_serializes_cleanly(self) -> None:
        answer = ResearchAnswer(
            answer="GraphRAG combines vector and graph retrieval.",
            confidence=ConfidenceLevel.MEDIUM,
            retrieval_strategy=RetrievalStrategy.HYBRID,
            evidence_ids=["evidence:aaaa"],
        )
        payload = _round_trips_through_plain_json(answer)
        assert payload["confidence"] == "medium"
