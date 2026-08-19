"""Unit tests for `EntityAliasRegistry` (prompt #15/#16/#60)."""

from app.domain.enums import EntityType
from app.ingestion.canonical_resolution.alias_registry import (
    EntityAliasRegistry,
    get_default_alias_registry,
)


class TestDefaultRegistry:
    def test_default_registry_is_empty(self) -> None:
        """V1 ships with zero asserted aliases (module docstring) -- a
        deliberate, conservative default, not an oversight."""

        registry = get_default_alias_registry()
        assert registry.resolve(EntityType.METHOD, "GraphRAG") is None
        assert registry.resolve(EntityType.DATASET, "MIMIC") is None


class TestExplicitAliasResolution:
    def test_configured_alias_resolves_to_canonical_name(self) -> None:
        registry = EntityAliasRegistry({(EntityType.METHOD, "graph rag"): "GraphRAG"})

        assert registry.resolve(EntityType.METHOD, "Graph RAG") == "GraphRAG"
        assert registry.resolve(EntityType.METHOD, "  graph   rag ") == "GraphRAG"

    def test_alias_is_scoped_by_entity_type(self) -> None:
        """The same normalized alias text under a different entity_type
        must not resolve (prompt #59: type separation)."""

        registry = EntityAliasRegistry({(EntityType.METHOD, "mimic"): "MIMIC-Method"})

        assert registry.resolve(EntityType.DATASET, "mimic") is None

    def test_unconfigured_name_resolves_to_none(self) -> None:
        registry = EntityAliasRegistry({(EntityType.METHOD, "graph rag"): "GraphRAG"})

        assert registry.resolve(EntityType.METHOD, "Microsoft GraphRAG") is None


class TestChecksum:
    def test_empty_registry_and_populated_registry_have_different_checksums(self) -> None:
        empty = EntityAliasRegistry({})
        populated = EntityAliasRegistry({(EntityType.METHOD, "graph rag"): "GraphRAG"})

        assert empty.checksum != populated.checksum

    def test_checksum_is_order_independent(self) -> None:
        a = EntityAliasRegistry(
            {
                (EntityType.METHOD, "graph rag"): "GraphRAG",
                (EntityType.DATASET, "mimic iv"): "MIMIC-IV",
            }
        )
        b = EntityAliasRegistry(
            {
                (EntityType.DATASET, "mimic iv"): "MIMIC-IV",
                (EntityType.METHOD, "graph rag"): "GraphRAG",
            }
        )

        assert a.checksum == b.checksum

    def test_changing_one_alias_changes_the_checksum(self) -> None:
        a = EntityAliasRegistry({(EntityType.METHOD, "graph rag"): "GraphRAG"})
        b = EntityAliasRegistry({(EntityType.METHOD, "graph rag"): "Graph-based RAG"})

        assert a.checksum != b.checksum
