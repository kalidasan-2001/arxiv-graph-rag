"""Integration tests for `Neo4jGraphRepository` against a real Neo4j
instance (prompt #71/#72) -- constraints, batched `MERGE` upserts,
generation-scoped read/delete, and paper/entity graph queries. Requires a
reachable Neo4j (see `tests/integration/conftest.py`); skipped
automatically otherwise.

Constraints/`MERGE` semantics/generation cleanup cannot be faithfully
validated with a mock (prompt #72) -- this file is the only place that
exercises real Cypher against a real database.
"""

import pytest
from neo4j import GraphDatabase

from app.core.exceptions import GraphStoreUnavailableError
from app.graph.models import GraphNodeInput, GraphRelationshipInput
from app.graph.neo4j_repository import Neo4jGraphRepository


def _node(entity_id: str, entity_type: str, canonical_name: str, **overrides) -> GraphNodeInput:
    defaults = dict(entity_id=entity_id, entity_type=entity_type, canonical_name=canonical_name)
    defaults.update(overrides)
    return GraphNodeInput(**defaults)


def _relationship(relationship_id: str, source: str, target: str, rel_type: str, **overrides) -> GraphRelationshipInput:
    defaults = dict(
        relationship_id=relationship_id,
        source_entity_id=source,
        target_entity_id=target,
        relationship_type=rel_type,
        confidence=0.9,
        provenance_type="chunk",
        paper_version_id="paper-version:arxiv:2401.90001:v1",
        graph_index_generation_fingerprint="gen-a",
    )
    defaults.update(overrides)
    return GraphRelationshipInput(**defaults)


class TestEnsureSchema:
    def test_is_idempotent(self, neo4j_repository) -> None:
        neo4j_repository.ensure_schema()
        neo4j_repository.ensure_schema()  # must not raise on the second call


class TestUpsertEntitiesConstraint:
    def test_merging_the_same_entity_id_twice_never_creates_a_duplicate_node(self, neo4j_repository) -> None:
        """Prompt #71: the uniqueness constraint (plus `MERGE` on
        `entity_id`) means re-upserting the same entity can never produce
        two nodes."""

        neo4j_repository.ensure_schema()
        node = _node("paper:arxiv:2401.90001", "paper", "Paper One", properties={"title": "Paper One"})

        neo4j_repository.upsert_entities([node])
        neo4j_repository.upsert_entities([node])

        with neo4j_repository._driver.session(database=neo4j_repository._database) as session:
            count = session.run(
                "MATCH (n:Paper {entity_id: $id}) RETURN count(n) AS c", id="paper:arxiv:2401.90001"
            ).single()["c"]
        assert count == 1

    def test_on_match_conservatively_unions_aliases_without_dropping_existing_ones(self, neo4j_repository) -> None:
        neo4j_repository.upsert_entities(
            [_node("entity:method:aaa", "method", "GraphRAG", aliases=["Graph-based RAG"])]
        )
        neo4j_repository.upsert_entities(
            [_node("entity:method:aaa", "method", "GraphRAG", aliases=["GraphRAG-lite"])]
        )

        entity = neo4j_repository.get_entity("entity:method:aaa")
        assert set(entity.aliases) == {"Graph-based RAG", "GraphRAG-lite"}

    def test_placeholder_paper_is_enriched_when_the_real_paper_is_later_indexed(self, neo4j_repository) -> None:
        """Prompt #40: enrichment must never downgrade -- but a
        placeholder's blank `title` is safely filled in once the same
        `entity_id` is written again with real data."""

        neo4j_repository.upsert_entities(
            [
                _node(
                    "paper:arxiv:2401.90002",
                    "paper",
                    "arxiv:2401.90002",
                    properties={"source": "arxiv", "source_id": "2401.90002", "is_placeholder": True},
                )
            ]
        )
        neo4j_repository.upsert_entities(
            [
                _node(
                    "paper:arxiv:2401.90002",
                    "paper",
                    "The Real Title",
                    properties={"source": "arxiv", "source_id": "2401.90002", "title": "The Real Title"},
                )
            ]
        )

        entity = neo4j_repository.get_entity("paper:arxiv:2401.90002")
        assert entity.properties["title"] == "The Real Title"


class TestUpsertRelationships:
    def test_relationship_is_merged_by_relationship_id_not_duplicated(self, neo4j_repository) -> None:
        neo4j_repository.upsert_entities(
            [
                _node("paper:arxiv:2401.90003", "paper", "Paper Three"),
                _node("entity:method:bbb", "method", "GraphRAG"),
            ]
        )
        rel = _relationship("rel-1", "paper:arxiv:2401.90003", "entity:method:bbb", "uses_method")

        neo4j_repository.upsert_relationships([rel])
        neo4j_repository.upsert_relationships([rel])

        with neo4j_repository._driver.session(database=neo4j_repository._database) as session:
            count = session.run(
                "MATCH ()-[r {relationship_id: $id}]->() RETURN count(r) AS c", id="rel-1"
            ).single()["c"]
        assert count == 1

    def test_relationship_endpoints_must_already_exist(self, neo4j_repository) -> None:
        """No entity was upserted first -- `MATCH` finds no endpoint, so
        `MERGE` writes nothing (the service's post-upsert verification is
        what turns this into a loud failure, not this layer)."""

        rel = _relationship("rel-missing", "paper:arxiv:missing", "entity:method:missing", "uses_method")
        neo4j_repository.upsert_relationships([rel])

        ids = neo4j_repository.get_relationship_ids_for_generation(
            "paper-version:arxiv:2401.90001:v1", generation_fingerprint="gen-a"
        )
        assert "rel-missing" not in ids


class TestGenerationQueries:
    def test_get_relationship_ids_for_generation_is_scoped_to_paper_and_fingerprint(self, neo4j_repository) -> None:
        neo4j_repository.upsert_entities(
            [
                _node("paper:arxiv:2401.90004", "paper", "Paper Four"),
                _node("entity:method:ccc", "method", "MethodC"),
                _node("entity:dataset:ddd", "dataset", "DatasetD"),
            ]
        )
        neo4j_repository.upsert_relationships(
            [
                _relationship(
                    "rel-gen-a",
                    "paper:arxiv:2401.90004",
                    "entity:method:ccc",
                    "uses_method",
                    paper_version_id="pv-4",
                    graph_index_generation_fingerprint="gen-a",
                ),
                _relationship(
                    "rel-gen-b",
                    "paper:arxiv:2401.90004",
                    "entity:dataset:ddd",
                    "evaluated_on",
                    paper_version_id="pv-4",
                    graph_index_generation_fingerprint="gen-b",
                ),
            ]
        )

        ids_a = neo4j_repository.get_relationship_ids_for_generation("pv-4", generation_fingerprint="gen-a")
        ids_b = neo4j_repository.get_relationship_ids_for_generation("pv-4", generation_fingerprint="gen-b")
        assert ids_a == {"rel-gen-a"}
        assert ids_b == {"rel-gen-b"}

    def test_delete_generation_removes_only_the_excluded_generation_for_that_paper_version(
        self, neo4j_repository
    ) -> None:
        neo4j_repository.upsert_entities(
            [
                _node("paper:arxiv:2401.90005", "paper", "Paper Five"),
                _node("entity:method:eee", "method", "MethodE"),
                _node("entity:method:fff", "method", "MethodF"),
            ]
        )
        neo4j_repository.upsert_relationships(
            [
                _relationship(
                    "rel-old",
                    "paper:arxiv:2401.90005",
                    "entity:method:eee",
                    "uses_method",
                    paper_version_id="pv-5",
                    graph_index_generation_fingerprint="gen-old",
                ),
                _relationship(
                    "rel-new",
                    "paper:arxiv:2401.90005",
                    "entity:method:fff",
                    "uses_method",
                    paper_version_id="pv-5",
                    graph_index_generation_fingerprint="gen-new",
                ),
            ]
        )

        deleted = neo4j_repository.delete_generation("pv-5", exclude_generation_fingerprint="gen-new")

        assert deleted == 1
        remaining = neo4j_repository.get_relationship_ids_for_generation(
            "pv-5", generation_fingerprint="gen-new"
        )
        assert remaining == {"rel-new"}
        # The stale relationship's *node* (MethodE) must still exist --
        # only the relationship is removed (prompt #45/#46/#47).
        assert neo4j_repository.get_entity("entity:method:eee") is not None

    def test_shared_node_survives_stale_cleanup_of_one_paper_while_another_still_references_it(
        self, neo4j_repository
    ) -> None:
        """Prompt #67: Method X shared by two papers -- reindexing Paper A
        away from Method X must not affect Paper B's still-current edge to
        the same node."""

        neo4j_repository.upsert_entities(
            [
                _node("paper:arxiv:2401.90006", "paper", "Paper A"),
                _node("paper:arxiv:2401.90007", "paper", "Paper B"),
                _node("entity:method:shared", "method", "SharedMethod"),
            ]
        )
        neo4j_repository.upsert_relationships(
            [
                _relationship(
                    "rel-a-old",
                    "paper:arxiv:2401.90006",
                    "entity:method:shared",
                    "uses_method",
                    paper_version_id="pv-a",
                    graph_index_generation_fingerprint="gen-a-old",
                ),
                _relationship(
                    "rel-b-current",
                    "paper:arxiv:2401.90007",
                    "entity:method:shared",
                    "uses_method",
                    paper_version_id="pv-b",
                    graph_index_generation_fingerprint="gen-b-current",
                ),
            ]
        )

        # Paper A reindexed away from the shared method -- only Paper A's
        # (pv-a) stale generation is cleaned up.
        deleted = neo4j_repository.delete_generation("pv-a", exclude_generation_fingerprint="gen-a-new")

        assert deleted == 1
        assert neo4j_repository.get_entity("entity:method:shared") is not None
        assert neo4j_repository.get_relationship_ids_for_generation(
            "pv-b", generation_fingerprint="gen-b-current"
        ) == {"rel-b-current"}


class TestQueries:
    def test_get_paper_graph_returns_directly_connected_nodes_and_relationships(self, neo4j_repository) -> None:
        neo4j_repository.upsert_entities(
            [
                _node("paper:arxiv:2401.90008", "paper", "Paper Eight", properties={"title": "Paper Eight"}),
                _node("entity:author:aaa", "author", "Jane Doe"),
                _node("entity:method:ggg", "method", "MethodG"),
            ]
        )
        neo4j_repository.upsert_relationships(
            [
                _relationship(
                    "rel-authored",
                    "paper:arxiv:2401.90008",
                    "entity:author:aaa",
                    "authored_by",
                    provenance_type="metadata",
                    graph_index_generation_fingerprint="gen-8",
                ),
                _relationship(
                    "rel-uses",
                    "paper:arxiv:2401.90008",
                    "entity:method:ggg",
                    "uses_method",
                    graph_index_generation_fingerprint="gen-8",
                ),
            ]
        )

        paper, nodes, relationships = neo4j_repository.get_paper_graph("paper:arxiv:2401.90008")

        assert paper is not None
        assert paper.properties["title"] == "Paper Eight"
        assert {n.entity_id for n in nodes} == {"entity:author:aaa", "entity:method:ggg"}
        assert {r.relationship_id for r in relationships} == {"rel-authored", "rel-uses"}
        authored = next(r for r in relationships if r.relationship_id == "rel-authored")
        assert authored.provenance_type == "metadata"

    def test_get_paper_graph_returns_none_for_an_unindexed_paper(self, neo4j_repository) -> None:
        paper, nodes, relationships = neo4j_repository.get_paper_graph("paper:arxiv:does-not-exist")
        assert paper is None
        assert nodes == []
        assert relationships == []

    def test_get_entity_context_includes_inbound_relationships(self, neo4j_repository) -> None:
        """Backs "which papers use Method X" (prompt #74) -- an inbound
        `USES_METHOD` edge onto the Method entity."""

        neo4j_repository.upsert_entities(
            [
                _node("paper:arxiv:2401.90009", "paper", "Paper Nine"),
                _node("paper:arxiv:2401.90010", "paper", "Paper Ten"),
                _node("entity:method:hhh", "method", "MethodH"),
            ]
        )
        neo4j_repository.upsert_relationships(
            [
                _relationship("rel-9", "paper:arxiv:2401.90009", "entity:method:hhh", "uses_method"),
                _relationship("rel-10", "paper:arxiv:2401.90010", "entity:method:hhh", "uses_method"),
            ]
        )

        context = neo4j_repository.get_entity_context("entity:method:hhh")

        assert context is not None
        entity, connected, relationships = context
        assert entity.entity_id == "entity:method:hhh"
        assert {n.entity_id for n in connected} == {"paper:arxiv:2401.90009", "paper:arxiv:2401.90010"}
        assert {r.relationship_id for r in relationships} == {"rel-9", "rel-10"}


class TestNeo4jUnavailable:
    def test_unreachable_neo4j_raises_a_typed_error(self) -> None:
        """Prompt #70: a typed failure, not a silent fallback or a crash
        with a raw driver exception leaking through."""

        driver = GraphDatabase.driver(
            "bolt://localhost:1", auth=("neo4j", "wrong"), connection_timeout=1
        )
        repository = Neo4jGraphRepository(driver, "neo4j")
        try:
            with pytest.raises(GraphStoreUnavailableError):
                repository.ensure_schema()
        finally:
            driver.close()
