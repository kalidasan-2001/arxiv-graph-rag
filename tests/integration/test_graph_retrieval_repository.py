"""Prompt 10 Neo4j integration tests for deterministic graph retrieval."""

from app.graph.models import GraphNodeInput, GraphRelationshipInput


def _node(entity_id: str, entity_type: str, canonical_name: str, **overrides) -> GraphNodeInput:
    defaults = dict(entity_id=entity_id, entity_type=entity_type, canonical_name=canonical_name)
    defaults.update(overrides)
    return GraphNodeInput(**defaults)


def _relationship(
    relationship_id: str,
    source: str,
    target: str,
    rel_type: str,
    *,
    confidence: float = 0.9,
    generation: str = "gen-current",
    **overrides,
) -> GraphRelationshipInput:
    defaults = dict(
        relationship_id=relationship_id,
        source_entity_id=source,
        target_entity_id=target,
        relationship_type=rel_type,
        confidence=confidence,
        extraction_version="v1",
        source_chunk_id=f"chunk:{relationship_id}",
        supporting_chunk_ids=[f"chunk:{relationship_id}"],
        provenance_type="chunk",
        paper_version_id="paper-version:arxiv:retrieval:v1",
        graph_index_generation_fingerprint=generation,
    )
    defaults.update(overrides)
    return GraphRelationshipInput(**defaults)


def _seed_graph(repository) -> None:
    repository.ensure_schema()
    repository.upsert_entities(
        [
            _node("paper:arxiv:a", "paper", "Paper A", properties={"title": "Paper A"}),
            _node("paper:arxiv:b", "paper", "Paper B", properties={"title": "Paper B"}),
            _node("paper:arxiv:c", "paper", "arxiv:paper-c", properties={"source": "arxiv", "source_id": "c", "is_placeholder": True}),
            _node("paper:arxiv:d", "paper", "Paper D", properties={"title": "Paper D"}),
            _node("entity:author:ada", "author", "Ada"),
            _node("entity:method:m1", "method", "Method One"),
            _node("entity:method:m2", "method", "Method Two"),
            _node("entity:dataset:d1", "dataset", "Dataset One"),
            _node("entity:dataset:d2", "dataset", "Dataset Two"),
            _node("entity:task:t1", "task", "Task One"),
        ]
    )
    repository.upsert_relationships(
        [
            _relationship("rel-author", "paper:arxiv:a", "entity:author:ada", "authored_by", confidence=1.0, provenance_type="metadata", source_chunk_id=None, supporting_chunk_ids=[]),
            _relationship("rel-a-m1", "paper:arxiv:a", "entity:method:m1", "uses_method", confidence=0.95),
            _relationship("rel-a-d1", "paper:arxiv:a", "entity:dataset:d1", "evaluated_on", confidence=0.91),
            _relationship("rel-a-t1", "paper:arxiv:a", "entity:task:t1", "addresses", confidence=0.88),
            _relationship("rel-a-c", "paper:arxiv:a", "paper:arxiv:c", "cites", confidence=1.0),
            _relationship("rel-c-d", "paper:arxiv:c", "paper:arxiv:d", "cites", confidence=1.0),
            _relationship("rel-b-cites-a", "paper:arxiv:b", "paper:arxiv:a", "cites", confidence=1.0),
            _relationship("rel-b-d1", "paper:arxiv:b", "entity:dataset:d1", "evaluated_on", confidence=0.87),
            _relationship("rel-b-m2", "paper:arxiv:b", "entity:method:m2", "uses_method", confidence=0.89),
            _relationship("rel-c-d2", "paper:arxiv:c", "entity:dataset:d2", "evaluated_on", confidence=0.86),
        ]
    )


class TestDirectGraphRetrieval:
    def test_one_hop_paper_primitives(self, neo4j_repository) -> None:
        _seed_graph(neo4j_repository)

        methods = neo4j_repository.get_direct_paths(
            "paper:arxiv:a", relationship_type="uses_method", direction="outgoing", end_entity_type="method"
        )
        datasets = neo4j_repository.get_direct_paths(
            "paper:arxiv:a", relationship_type="evaluated_on", direction="outgoing", end_entity_type="dataset"
        )
        tasks = neo4j_repository.get_direct_paths(
            "paper:arxiv:a", relationship_type="addresses", direction="outgoing", end_entity_type="task"
        )
        authors = neo4j_repository.get_direct_paths(
            "paper:arxiv:a", relationship_type="authored_by", direction="outgoing", end_entity_type="author"
        )

        assert methods[0].nodes[-1].entity_id == "entity:method:m1"
        assert datasets[0].nodes[-1].entity_id == "entity:dataset:d1"
        assert tasks[0].nodes[-1].entity_id == "entity:task:t1"
        assert authors[0].relationships[0].provenance_type == "metadata"
        assert authors[0].relationships[0].source_chunk_id is None

    def test_citations_and_inverse_entity_lookup(self, neo4j_repository) -> None:
        _seed_graph(neo4j_repository)

        citations = neo4j_repository.get_direct_paths(
            "paper:arxiv:a", relationship_type="cites", direction="outgoing", end_entity_type="paper"
        )
        cited_by = neo4j_repository.get_direct_paths(
            "paper:arxiv:a", relationship_type="cites", direction="incoming", end_entity_type="paper"
        )
        papers_for_dataset = neo4j_repository.get_direct_paths(
            "entity:dataset:d1", relationship_type="evaluated_on", direction="incoming", end_entity_type="paper"
        )

        assert citations[0].nodes[-1].properties["is_placeholder"] is True
        assert cited_by[0].nodes[-1].entity_id == "paper:arxiv:b"
        assert {path.nodes[-1].entity_id for path in papers_for_dataset} == {"paper:arxiv:a", "paper:arxiv:b"}
        assert citations[0].relationships[0].graph_index_generation_fingerprint == "gen-current"


class TestMultiHopGraphRetrieval:
    def test_shared_dataset_paths_exclude_the_start_paper(self, neo4j_repository) -> None:
        _seed_graph(neo4j_repository)

        paths = neo4j_repository.get_shared_entity_paths(
            "paper:arxiv:a",
            relationship_type="evaluated_on",
            shared_entity_type="dataset",
        )

        assert len(paths) == 1
        assert [node.entity_id for node in paths[0].nodes] == [
            "paper:arxiv:a",
            "entity:dataset:d1",
            "paper:arxiv:b",
        ]

    def test_shared_method_paths(self, neo4j_repository) -> None:
        _seed_graph(neo4j_repository)
        neo4j_repository.upsert_relationships(
            [_relationship("rel-b-m1", "paper:arxiv:b", "entity:method:m1", "uses_method")]
        )

        paths = neo4j_repository.get_shared_entity_paths(
            "paper:arxiv:a",
            relationship_type="uses_method",
            shared_entity_type="method",
        )

        assert len(paths) == 1
        assert paths[0].nodes[1].entity_id == "entity:method:m1"
        assert paths[0].nodes[2].entity_id == "paper:arxiv:b"

    def test_datasets_from_citing_papers(self, neo4j_repository) -> None:
        _seed_graph(neo4j_repository)

        paths = neo4j_repository.get_citing_paper_entity_paths(
            "paper:arxiv:a",
            relationship_type="evaluated_on",
            end_entity_type="dataset",
        )

        assert len(paths) == 1
        assert [node.entity_id for node in paths[0].nodes] == [
            "paper:arxiv:a",
            "paper:arxiv:b",
            "entity:dataset:d1",
        ]
        assert [rel.relationship_id for rel in paths[0].relationships] == ["rel-b-cites-a", "rel-b-d1"]

    def test_methods_for_dataset(self, neo4j_repository) -> None:
        _seed_graph(neo4j_repository)

        paths = neo4j_repository.get_entity_paper_entity_paths(
            "entity:dataset:d1",
            incoming_relationship_type="evaluated_on",
            outgoing_relationship_type="uses_method",
            end_entity_type="method",
        )

        assert {path.nodes[-1].entity_id for path in paths} == {"entity:method:m1", "entity:method:m2"}
        assert all(len(path.relationships) == 2 for path in paths)

    def test_citation_neighborhood_is_bounded(self, neo4j_repository) -> None:
        _seed_graph(neo4j_repository)

        paths = neo4j_repository.get_citation_neighborhood_paths(
            "paper:arxiv:a", depth=1, limit=10
        )

        assert {path.nodes[-1].entity_id for path in paths} == {"paper:arxiv:b", "paper:arxiv:c"}

    def test_citation_neighborhood_depth_two_preserves_edge_direction(self, neo4j_repository) -> None:
        _seed_graph(neo4j_repository)

        paths = neo4j_repository.get_citation_neighborhood_paths(
            "paper:arxiv:a", depth=2, limit=10
        )

        outgoing_path = next(path for path in paths if path.nodes[-1].entity_id == "paper:arxiv:d")
        assert [node.entity_id for node in outgoing_path.nodes] == [
            "paper:arxiv:a",
            "paper:arxiv:c",
            "paper:arxiv:d",
        ]
        assert [
            (rel.relationship_id, rel.source_entity_id, rel.target_entity_id)
            for rel in outgoing_path.relationships
        ] == [
            ("rel-a-c", "paper:arxiv:a", "paper:arxiv:c"),
            ("rel-c-d", "paper:arxiv:c", "paper:arxiv:d"),
        ]


class TestNameLookup:
    def test_name_lookup_can_be_scoped_by_entity_type(self, neo4j_repository) -> None:
        _seed_graph(neo4j_repository)

        matches = neo4j_repository.find_entities_by_canonical_name(
            "Dataset One", entity_type="dataset"
        )

        assert [match.entity_id for match in matches] == ["entity:dataset:d1"]
