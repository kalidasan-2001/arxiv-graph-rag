"""Unit tests for `CanonicalEntityResolver` (prompt #57-61, #66)."""

from app.domain.enums import EntityType, RelationshipType
from app.domain.knowledge import ScientificEntity, ScientificRelationship
from app.ingestion.canonical_resolution.alias_registry import EntityAliasRegistry
from app.ingestion.canonical_resolution.models import ResolutionTier
from app.ingestion.canonical_resolution.resolver import CanonicalEntityResolver


def _resolver(aliases: dict | None = None) -> CanonicalEntityResolver:
    return CanonicalEntityResolver(EntityAliasRegistry(aliases or {}))


class TestNormalization:
    def test_whitespace_and_case_variants_resolve_to_the_same_entity(self) -> None:
        """Prompt #57: `" GraphRAG "` and `"GraphRAG"` are Tier 2 --
        already-existing literal-formatting variants of the same string."""

        resolver = _resolver()
        a = resolver.resolve_entity(
            ScientificEntity.create(entity_type=EntityType.METHOD, canonical_name=" GraphRAG ")
        )
        b = resolver.resolve_entity(
            ScientificEntity.create(entity_type=EntityType.METHOD, canonical_name="GraphRAG")
        )

        assert a.canonical_entity.entity_id == b.canonical_entity.entity_id
        assert a.tier == ResolutionTier.EXACT_NORMALIZED
        assert b.tier == ResolutionTier.EXACT_NORMALIZED


class TestNoFuzzyOvermerge:
    def test_mimic_and_mimic_iv_stay_distinct_by_default(self) -> None:
        """Prompt #58: the default V1 resolver must produce two nodes for
        MIMIC/MIMIC-IV -- no fuzzy matching."""

        resolver = _resolver()
        mimic = resolver.resolve_entity(
            ScientificEntity.create(entity_type=EntityType.DATASET, canonical_name="MIMIC")
        )
        mimic_iv = resolver.resolve_entity(
            ScientificEntity.create(entity_type=EntityType.DATASET, canonical_name="MIMIC-IV")
        )

        assert mimic.canonical_entity.entity_id != mimic_iv.canonical_entity.entity_id

    def test_graphrag_and_microsoft_graphrag_stay_distinct_without_an_alias(self) -> None:
        resolver = _resolver()
        graphrag = resolver.resolve_entity(
            ScientificEntity.create(entity_type=EntityType.METHOD, canonical_name="GraphRAG")
        )
        microsoft_graphrag = resolver.resolve_entity(
            ScientificEntity.create(entity_type=EntityType.METHOD, canonical_name="Microsoft GraphRAG")
        )

        assert graphrag.canonical_entity.entity_id != microsoft_graphrag.canonical_entity.entity_id


class TestEntityTypeSeparation:
    def test_same_normalized_name_under_different_types_produces_different_ids(self) -> None:
        """Prompt #59: `METHOD: MIMIC` != `DATASET: MIMIC`."""

        resolver = _resolver()
        method = resolver.resolve_entity(
            ScientificEntity.create(entity_type=EntityType.METHOD, canonical_name="MIMIC")
        )
        dataset = resolver.resolve_entity(
            ScientificEntity.create(entity_type=EntityType.DATASET, canonical_name="MIMIC")
        )

        assert method.canonical_entity.entity_id != dataset.canonical_entity.entity_id


class TestExplicitAlias:
    def test_configured_alias_resolves_two_surface_forms_to_one_entity(self) -> None:
        """Prompt #43: only an explicit alias entry may merge two
        differently-worded surface forms."""

        resolver = _resolver({(EntityType.METHOD, "graph rag"): "GraphRAG"})
        graphrag = resolver.resolve_entity(
            ScientificEntity.create(entity_type=EntityType.METHOD, canonical_name="GraphRAG")
        )
        graph_rag = resolver.resolve_entity(
            ScientificEntity.create(entity_type=EntityType.METHOD, canonical_name="Graph RAG")
        )

        assert graph_rag.canonical_entity.entity_id == graphrag.canonical_entity.entity_id
        assert graph_rag.tier == ResolutionTier.EXPLICIT_ALIAS
        assert "Graph RAG" in graph_rag.canonical_entity.aliases


class TestPaperIdentity:
    def test_paper_entity_id_is_preserved_unchanged(self) -> None:
        """Prompt #61: a Paper's `entity_id` is always its own `paper_id` --
        never re-derived, never merged by title."""

        resolver = _resolver()
        paper_entity = ScientificEntity(
            entity_id="paper:arxiv:2401.99999",
            entity_type=EntityType.PAPER,
            canonical_name="An Unrelated-Looking Title",
        )

        resolution = resolver.resolve_entity(paper_entity)

        assert resolution.canonical_entity.entity_id == "paper:arxiv:2401.99999"
        assert resolution.tier == ResolutionTier.TRUSTED_IDENTITY

    def test_two_different_papers_never_merge_even_with_similar_titles(self) -> None:
        resolver = _resolver()
        a = resolver.resolve_entity(
            ScientificEntity(
                entity_id="paper:arxiv:2401.00001", entity_type=EntityType.PAPER, canonical_name="GraphRAG"
            )
        )
        b = resolver.resolve_entity(
            ScientificEntity(
                entity_id="paper:arxiv:2401.00002", entity_type=EntityType.PAPER, canonical_name="GraphRAG"
            )
        )

        assert a.canonical_entity.entity_id != b.canonical_entity.entity_id


class TestAuthorIdentity:
    def test_author_resolution_is_trusted_identity_tier(self) -> None:
        resolver = _resolver()
        resolution = resolver.resolve_entity(
            ScientificEntity.create(entity_type=EntityType.AUTHOR, canonical_name="Jane Doe")
        )
        assert resolution.tier == ResolutionTier.TRUSTED_IDENTITY


class TestBuildCanonicalGraph:
    def test_cross_paper_alias_merge_produces_one_entity_and_two_relationships(self) -> None:
        """Prompt #66: two papers' `USES_METHOD -> GraphRAG` candidates,
        spelled differently, resolve to the same canonical Method node
        when an explicit alias supports it -- and each paper still has its
        own `USES_METHOD` edge."""

        resolver = _resolver({(EntityType.METHOD, "graph rag"): "GraphRAG"})

        paper_a = ScientificEntity(
            entity_id="paper:arxiv:2401.00001", entity_type=EntityType.PAPER, canonical_name="Paper A"
        )
        method_a = ScientificEntity.create(entity_type=EntityType.METHOD, canonical_name="GraphRAG")
        relationship_a = ScientificRelationship.create(
            source_entity_id=paper_a.entity_id,
            relationship_type=RelationshipType.USES_METHOD,
            target_entity_id=method_a.entity_id,
            confidence=0.9,
            source_chunk_id="chunk:aaa",
        )

        graph_a = resolver.build_canonical_graph(
            paper_id=paper_a.entity_id,
            paper_version_id="paper-version:arxiv:2401.00001:v1",
            entities=[paper_a, method_a],
            relationships=[relationship_a],
            canonicalization_version="v1",
            canonicalization_config_fingerprint="fp-a",
            graph_index_generation_fingerprint="gen-a",
        )

        paper_b = ScientificEntity(
            entity_id="paper:arxiv:2401.00002", entity_type=EntityType.PAPER, canonical_name="Paper B"
        )
        method_b = ScientificEntity.create(entity_type=EntityType.METHOD, canonical_name="Graph RAG")
        relationship_b = ScientificRelationship.create(
            source_entity_id=paper_b.entity_id,
            relationship_type=RelationshipType.USES_METHOD,
            target_entity_id=method_b.entity_id,
            confidence=0.8,
            source_chunk_id="chunk:bbb",
        )

        graph_b = resolver.build_canonical_graph(
            paper_id=paper_b.entity_id,
            paper_version_id="paper-version:arxiv:2401.00002:v1",
            entities=[paper_b, method_b],
            relationships=[relationship_b],
            canonicalization_version="v1",
            canonicalization_config_fingerprint="fp-a",
            graph_index_generation_fingerprint="gen-a",
        )

        canonical_method_a = next(e for e in graph_a.entities if e.entity_type == EntityType.METHOD)
        canonical_method_b = next(e for e in graph_b.entities if e.entity_type == EntityType.METHOD)
        assert canonical_method_a.entity_id == canonical_method_b.entity_id
        assert graph_a.relationships[0].target_entity_id == canonical_method_a.entity_id
        assert graph_b.relationships[0].target_entity_id == canonical_method_b.entity_id
        assert graph_a.relationships[0].relationship_id != graph_b.relationships[0].relationship_id

    def test_relationship_ids_are_rebuilt_after_alias_remapping(self) -> None:
        """A relationship whose target got remapped by an alias must get a
        freshly-derived `relationship_id`, not the pre-remap one."""

        resolver = _resolver({(EntityType.DATASET, "mimic four"): "MIMIC-IV"})
        paper = ScientificEntity(
            entity_id="paper:arxiv:2401.00003", entity_type=EntityType.PAPER, canonical_name="Paper C"
        )
        raw_dataset = ScientificEntity.create(entity_type=EntityType.DATASET, canonical_name="Mimic Four")
        relationship = ScientificRelationship.create(
            source_entity_id=paper.entity_id,
            relationship_type=RelationshipType.EVALUATED_ON,
            target_entity_id=raw_dataset.entity_id,
            confidence=0.7,
            source_chunk_id="chunk:ccc",
        )

        graph = resolver.build_canonical_graph(
            paper_id=paper.entity_id,
            paper_version_id="paper-version:arxiv:2401.00003:v1",
            entities=[paper, raw_dataset],
            relationships=[relationship],
            canonicalization_version="v1",
            canonicalization_config_fingerprint="fp",
            graph_index_generation_fingerprint="gen",
        )

        canonical_dataset = next(e for e in graph.entities if e.entity_type == EntityType.DATASET)
        assert canonical_dataset.canonical_name == "MIMIC-IV"
        assert graph.relationships[0].target_entity_id == canonical_dataset.entity_id
        assert graph.relationships[0].relationship_id != relationship.relationship_id
