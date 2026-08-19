"""Unit tests for `app.ingestion.graph_indexing.graph_mapping` (prompt
#21-24, #64)."""

from app.domain.enums import EntityType, RelationshipType
from app.domain.knowledge import ScientificEntity, ScientificRelationship
from app.ingestion.graph_indexing.graph_mapping import (
    entity_to_node_input,
    relationship_to_relationship_input,
)


class TestEntityToNodeInput:
    def test_real_paper_entity_gets_a_title_property(self) -> None:
        entity = ScientificEntity(
            entity_id="paper:arxiv:2401.00001",
            entity_type=EntityType.PAPER,
            canonical_name="GraphSteal: Something",
            metadata={"source": "arxiv", "source_id": "2401.00001"},
        )

        node = entity_to_node_input(entity)

        assert node.properties["title"] == "GraphSteal: Something"
        assert node.properties["source"] == "arxiv"
        assert "is_placeholder" not in node.properties

    def test_placeholder_paper_entity_never_gets_a_title(self) -> None:
        """Prompt #19: a reference-only citation-target node must never be
        given a fabricated title."""

        entity = ScientificEntity(
            entity_id="paper:arxiv:2401.00002",
            entity_type=EntityType.PAPER,
            canonical_name="arxiv:2401.00002",
            metadata={"source": "arxiv", "source_id": "2401.00002", "placeholder": True},
        )

        node = entity_to_node_input(entity)

        assert "title" not in node.properties
        assert node.properties["is_placeholder"] is True

    def test_non_paper_entity_gets_no_extra_properties(self) -> None:
        entity = ScientificEntity.create(entity_type=EntityType.METHOD, canonical_name="GraphRAG")

        node = entity_to_node_input(entity)

        assert node.properties == {}
        assert node.entity_type == "method"
        assert node.canonical_name == "GraphRAG"


class TestRelationshipToRelationshipInput:
    def test_authored_by_gets_metadata_provenance(self) -> None:
        """Prompt #64: `AUTHORED_BY` is created from arXiv metadata, not a
        chunk -- it must never be given a fake chunk provenance."""

        relationship = ScientificRelationship.create(
            source_entity_id="paper:arxiv:2401.00001",
            relationship_type=RelationshipType.AUTHORED_BY,
            target_entity_id="entity:author:aaaaaaaaaaaaaaaa",
            confidence=1.0,
            metadata={"resolution": "arxiv_metadata"},
        )

        rel_input = relationship_to_relationship_input(
            relationship, paper_version_id="paper-version:arxiv:2401.00001:v1", graph_index_generation_fingerprint="gen-1"
        )

        assert rel_input.provenance_type == "metadata"
        assert rel_input.source_chunk_id is None

    def test_uses_method_gets_chunk_provenance(self) -> None:
        relationship = ScientificRelationship.create(
            source_entity_id="paper:arxiv:2401.00001",
            relationship_type=RelationshipType.USES_METHOD,
            target_entity_id="entity:method:bbbbbbbbbbbbbbbb",
            confidence=0.85,
            source_chunk_id="chunk:xyz",
            extraction_version="v1",
            metadata={"supporting_chunk_ids": ["chunk:xyz", "chunk:abc"]},
        )

        rel_input = relationship_to_relationship_input(
            relationship, paper_version_id="paper-version:arxiv:2401.00001:v1", graph_index_generation_fingerprint="gen-1"
        )

        assert rel_input.provenance_type == "chunk"
        assert rel_input.source_chunk_id == "chunk:xyz"
        assert rel_input.supporting_chunk_ids == ["chunk:xyz", "chunk:abc"]
        assert rel_input.confidence == 0.85
        assert rel_input.extraction_version == "v1"
        assert rel_input.graph_index_generation_fingerprint == "gen-1"
        assert rel_input.paper_version_id == "paper-version:arxiv:2401.00001:v1"
