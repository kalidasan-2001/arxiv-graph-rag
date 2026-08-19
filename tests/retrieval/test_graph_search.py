from app.core.exceptions import GraphEntityAmbiguousError, GraphEntityNotFoundError, GraphSearchError
from app.domain.enums import EntityType, EvidenceScoreKind, EvidenceSourceStore
from app.graph.models import GraphNodeRecord, GraphPathRecord, GraphRelationshipRecord
from app.retrieval.graph_search import GraphRetrievalService, GraphSearchOperation


def _node(entity_id: str, entity_type: str, name: str) -> GraphNodeRecord:
    return GraphNodeRecord(entity_id=entity_id, entity_type=entity_type, canonical_name=name)


def _rel(
    relationship_id: str,
    source: str,
    target: str,
    relationship_type: str = "uses_method",
    *,
    confidence: float = 0.9,
) -> GraphRelationshipRecord:
    return GraphRelationshipRecord(
        relationship_id=relationship_id,
        source_entity_id=source,
        target_entity_id=target,
        relationship_type=relationship_type,
        confidence=confidence,
        extraction_version="v1",
        source_chunk_id=f"chunk:{relationship_id}",
        supporting_chunk_ids=[f"chunk:{relationship_id}"],
        provenance_type="chunk",
        paper_version_id="paper-version:arxiv:1:v1",
        graph_index_generation_fingerprint="gen-1",
    )


class FakeGraphRepository:
    def __init__(self) -> None:
        self.nodes: dict[str, GraphNodeRecord] = {}
        self.name_candidates: list[GraphNodeRecord] = []
        self.paths: list[GraphPathRecord] = []

    def get_entity(self, entity_id: str):
        return self.nodes.get(entity_id)

    def find_entities_by_canonical_name(self, canonical_name: str, *, entity_type=None, limit=20):
        return self.name_candidates[:limit]

    def get_direct_paths(self, *args, **kwargs):
        return self.paths

    def get_shared_entity_paths(self, *args, **kwargs):
        return self.paths

    def get_citing_paper_entity_paths(self, *args, **kwargs):
        return self.paths

    def get_entity_paper_entity_paths(self, *args, **kwargs):
        return self.paths

    def get_citation_neighborhood_paths(self, *args, **kwargs):
        return self.paths


def _service(repo: FakeGraphRepository) -> GraphRetrievalService:
    return GraphRetrievalService(repo, max_depth=3, default_limit=20, max_limit=100)


class TestGraphRetrievalServiceValidation:
    def test_depth_must_be_bounded(self) -> None:
        repo = FakeGraphRepository()
        repo.nodes["paper:arxiv:1"] = _node("paper:arxiv:1", "paper", "Paper")

        try:
            _service(repo).search(
                operation=GraphSearchOperation.PAPER_METHODS,
                entity_id="paper:arxiv:1",
                depth=4,
            )
        except GraphSearchError as exc:
            assert "depth" in str(exc)
        else:
            raise AssertionError("expected GraphSearchError")

    def test_limit_must_be_bounded(self) -> None:
        repo = FakeGraphRepository()
        repo.nodes["paper:arxiv:1"] = _node("paper:arxiv:1", "paper", "Paper")

        try:
            _service(repo).search(
                operation=GraphSearchOperation.PAPER_METHODS,
                entity_id="paper:arxiv:1",
                limit=0,
            )
        except GraphSearchError as exc:
            assert "limit" in str(exc)
        else:
            raise AssertionError("expected GraphSearchError")

    def test_missing_entity_is_typed(self) -> None:
        try:
            _service(FakeGraphRepository()).search(
                operation=GraphSearchOperation.PAPER_METHODS,
                entity_id="paper:arxiv:missing",
            )
        except GraphEntityNotFoundError:
            pass
        else:
            raise AssertionError("expected GraphEntityNotFoundError")

    def test_ambiguous_name_lookup_is_typed_and_exposes_candidates(self) -> None:
        repo = FakeGraphRepository()
        repo.name_candidates = [
            _node("entity:method:1", "method", "MIMIC"),
            _node("entity:dataset:1", "dataset", "MIMIC"),
        ]

        try:
            _service(repo).search(
                operation=GraphSearchOperation.PAPERS_FOR_DATASET,
                canonical_name="MIMIC",
            )
        except GraphEntityAmbiguousError as exc:
            assert {candidate["entity_type"] for candidate in exc.candidates} == {"method", "dataset"}
        else:
            raise AssertionError("expected GraphEntityAmbiguousError")


class TestGraphEvidenceNormalization:
    def test_graph_relationship_evidence_preserves_provenance_and_stable_ids(self) -> None:
        repo = FakeGraphRepository()
        paper = _node("paper:arxiv:1", "paper", "Paper")
        method = _node("entity:method:1", "method", "GraphRAG")
        repo.nodes[paper.entity_id] = paper
        repo.paths = [
            GraphPathRecord(
                nodes=[paper, method],
                relationships=[_rel("rel-1", paper.entity_id, method.entity_id)],
            )
        ]

        first = _service(repo).search(
            operation=GraphSearchOperation.PAPER_METHODS, entity_id=paper.entity_id
        )
        second = _service(repo).search(
            operation=GraphSearchOperation.PAPER_METHODS, entity_id=paper.entity_id
        )

        evidence = first.evidence[0]
        assert evidence.evidence_id == second.evidence[0].evidence_id
        assert evidence.evidence_type.value == "graph_relationship"
        assert evidence.relationship_ids == ["rel-1"]
        assert evidence.metadata["relationships"][0]["graph_index_generation_fingerprint"] == "gen-1"
        assert evidence.metadata["source_chunk_ids"] == ["chunk:rel-1"]
        assert evidence.metadata["path_confidence"] == 0.9
        assert evidence.score == 0.9
        assert evidence.score_kind == EvidenceScoreKind.GRAPH_PATH_CONFIDENCE
        assert evidence.source_store == EvidenceSourceStore.NEO4J
        assert evidence.provenance.graph_index_generation_fingerprint == "gen-1"

    def test_path_evidence_preserves_ordered_path(self) -> None:
        repo = FakeGraphRepository()
        dataset = _node("entity:dataset:1", "dataset", "MIMIC-IV")
        paper = _node("paper:arxiv:1", "paper", "Paper")
        method = _node("entity:method:1", "method", "GraphRAG")
        repo.nodes[dataset.entity_id] = dataset
        repo.paths = [
            GraphPathRecord(
                nodes=[dataset, paper, method],
                relationships=[
                    _rel("rel-data", paper.entity_id, dataset.entity_id, "evaluated_on", confidence=0.7),
                    _rel("rel-method", paper.entity_id, method.entity_id, "uses_method", confidence=0.95),
                ],
            )
        ]

        result = _service(repo).search(
            operation=GraphSearchOperation.METHODS_FOR_DATASET,
            entity_id=dataset.entity_id,
        )

        evidence = result.evidence[0]
        assert evidence.evidence_type.value == "graph_path"
        assert evidence.metadata["ordered_entity_ids"] == [
            "entity:dataset:1",
            "paper:arxiv:1",
            "entity:method:1",
        ]
        assert evidence.metadata["ordered_relationship_ids"] == ["rel-data", "rel-method"]
        assert evidence.metadata["path_confidence"] == 0.7

    def test_ranking_prefers_shorter_then_higher_confidence_then_stable_id(self) -> None:
        repo = FakeGraphRepository()
        paper = _node("paper:arxiv:1", "paper", "Paper")
        method_a = _node("entity:method:a", "method", "A")
        method_b = _node("entity:method:b", "method", "B")
        repo.nodes[paper.entity_id] = paper
        repo.paths = [
            GraphPathRecord(nodes=[paper, method_a], relationships=[_rel("rel-low", paper.entity_id, method_a.entity_id, confidence=0.2)]),
            GraphPathRecord(nodes=[paper, method_b], relationships=[_rel("rel-high", paper.entity_id, method_b.entity_id, confidence=0.9)]),
        ]

        result = _service(repo).search(
            operation=GraphSearchOperation.PAPER_METHODS, entity_id=paper.entity_id
        )

        assert [item.path_confidence for item in result.results] == [0.9, 0.2]

    def test_name_resolution_uses_canonical_resolver_policy(self) -> None:
        repo = FakeGraphRepository()
        # Graph RAG remains distinct from GraphRAG under the default V1
        # resolver because no explicit alias maps it.
        graphrag = _node("entity:method:858edf720f19ac91", "method", "GraphRAG")
        repo.nodes[graphrag.entity_id] = graphrag

        result = _service(repo).search(
            operation=GraphSearchOperation.PAPERS_FOR_METHOD,
            entity_type=EntityType.METHOD,
            canonical_name="GraphRAG",
        )

        assert result.start_entity.entity_id == "entity:method:858edf720f19ac91"
