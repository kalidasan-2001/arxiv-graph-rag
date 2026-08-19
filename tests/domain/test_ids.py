"""Determinism tests for the id-generation helpers in `app.domain.ids`."""

from app.domain.enums import EntityType, EvidenceType, RelationshipType, SectionType
from app.domain.ids import (
    build_chunk_id,
    build_entity_id,
    build_evidence_id,
    build_ingestion_job_id,
    build_paper_id,
    build_paper_version_id,
    build_relationship_id,
    build_section_id,
)


class TestEntityIdDeterminism:
    def test_same_normalized_name_yields_same_id(self) -> None:
        first = build_entity_id(EntityType.METHOD, "GraphRAG")
        second = build_entity_id(EntityType.METHOD, "GraphRAG")
        assert first == second

    def test_whitespace_variations_yield_same_id(self) -> None:
        first = build_entity_id(EntityType.METHOD, "GraphRAG")
        assert first == build_entity_id(EntityType.METHOD, "  GraphRAG  ")
        assert first == build_entity_id(EntityType.METHOD, "graphrag")

    def test_case_insensitive_but_meaning_preserving(self) -> None:
        assert build_entity_id(EntityType.METHOD, "Graph  RAG") == build_entity_id(
            EntityType.METHOD, "graph rag"
        )

    def test_different_entity_types_same_name_do_not_collide(self) -> None:
        method_id = build_entity_id(EntityType.METHOD, "BERT")
        dataset_id = build_entity_id(EntityType.DATASET, "BERT")
        assert method_id != dataset_id

    def test_different_names_do_not_collide(self) -> None:
        assert build_entity_id(EntityType.METHOD, "BERT") != build_entity_id(
            EntityType.METHOD, "RoBERTa"
        )


class TestPaperIdentity:
    def test_same_source_id_yields_same_paper_id(self) -> None:
        first = build_paper_id("arxiv", "2401.12345")
        second = build_paper_id("arxiv", "2401.12345")
        assert first == second

    def test_versions_share_paper_id_but_differ_in_version_id(self) -> None:
        paper_id = build_paper_id("arxiv", "2401.12345")
        v1 = build_paper_version_id(paper_id, "1")
        v2 = build_paper_version_id(paper_id, "2")

        assert v1 != v2
        # Both versions still resolve back to the same logical paper.
        assert v1.startswith("paper-version:arxiv:2401.12345:")
        assert v2.startswith("paper-version:arxiv:2401.12345:")


class TestChunkIdentity:
    def _chunk_id(
        self, *, paper_version_id: str = "paper-version:arxiv:2401.12345:v1",
        section_id: str = "section:aaaa", chunk_index: int = 0,
        chunk_config_fingerprint: str = "fp-a",
    ) -> str:
        return build_chunk_id(paper_version_id, section_id, chunk_index, chunk_config_fingerprint)

    def test_identical_inputs_produce_identical_id(self) -> None:
        assert self._chunk_id() == self._chunk_id()

    def test_different_paper_version_changes_id(self) -> None:
        assert self._chunk_id() != self._chunk_id(
            paper_version_id="paper-version:arxiv:2401.12345:v2"
        )

    def test_different_section_changes_id(self) -> None:
        assert self._chunk_id() != self._chunk_id(section_id="section:bbbb")

    def test_different_chunk_index_changes_id(self) -> None:
        assert self._chunk_id() != self._chunk_id(chunk_index=1)

    def test_different_config_fingerprint_changes_id(self) -> None:
        # Critical for re-chunking safety (prompt 6.1/#8): a different
        # effective chunking configuration -- represented by its
        # fingerprint, not a bare version string -- must never collide
        # with a previous run's ids.
        assert self._chunk_id() != self._chunk_id(chunk_config_fingerprint="fp-b")


class TestSectionIdentity:
    def test_identical_inputs_produce_identical_id(self) -> None:
        paper_version_id = "paper-version:arxiv:2401.12345:v1"
        first = build_section_id(paper_version_id, SectionType.METHODOLOGY, 2)
        second = build_section_id(paper_version_id, SectionType.METHODOLOGY, 2)
        assert first == second

    def test_different_order_changes_id(self) -> None:
        paper_version_id = "paper-version:arxiv:2401.12345:v1"
        first = build_section_id(paper_version_id, SectionType.METHODOLOGY, 2)
        second = build_section_id(paper_version_id, SectionType.METHODOLOGY, 3)
        assert first != second


class TestRelationshipIdentity:
    def test_same_triple_yields_same_id(self) -> None:
        source = build_entity_id(EntityType.PAPER, "Paper A")
        target = build_entity_id(EntityType.METHOD, "GraphRAG")
        first = build_relationship_id(source, RelationshipType.USES_METHOD, target)
        second = build_relationship_id(source, RelationshipType.USES_METHOD, target)
        assert first == second

    def test_different_relationship_types_do_not_collide(self) -> None:
        source = build_entity_id(EntityType.PAPER, "Paper A")
        target = build_entity_id(EntityType.METHOD, "GraphRAG")
        uses_method = build_relationship_id(source, RelationshipType.USES_METHOD, target)
        evaluated_on = build_relationship_id(source, RelationshipType.EVALUATED_ON, target)
        assert uses_method != evaluated_on

    def test_swapped_source_and_target_do_not_collide(self) -> None:
        a = build_entity_id(EntityType.PAPER, "Paper A")
        b = build_entity_id(EntityType.METHOD, "GraphRAG")
        forward = build_relationship_id(a, RelationshipType.USES_METHOD, b)
        backward = build_relationship_id(b, RelationshipType.USES_METHOD, a)
        assert forward != backward


class TestEvidenceIdentity:
    def test_same_references_yield_same_id(self) -> None:
        first = build_evidence_id(EvidenceType.TEXT, "chunk:aaaa")
        second = build_evidence_id(EvidenceType.TEXT, "chunk:aaaa")
        assert first == second

    def test_different_evidence_type_changes_id(self) -> None:
        text = build_evidence_id(EvidenceType.TEXT, "chunk:aaaa")
        graph = build_evidence_id(EvidenceType.GRAPH_RELATIONSHIP, "chunk:aaaa")
        assert text != graph


class TestIngestionJobId:
    def test_ingestion_job_ids_are_unique_per_call(self) -> None:
        # Ingestion jobs are operational runs, not stable content -- two
        # calls (e.g. two retries of the same paper) must not collide.
        assert build_ingestion_job_id() != build_ingestion_job_id()
