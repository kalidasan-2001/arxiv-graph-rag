from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.graph.models import GraphNodeRecord, GraphPathRecord, GraphRelationshipRecord
from app.graph.neo4j_repository import get_graph_repository
from app.main import app


def _node(entity_id: str, entity_type: str, name: str) -> GraphNodeRecord:
    return GraphNodeRecord(entity_id=entity_id, entity_type=entity_type, canonical_name=name)


def _rel(relationship_id: str, source: str, target: str) -> GraphRelationshipRecord:
    return GraphRelationshipRecord(
        relationship_id=relationship_id,
        source_entity_id=source,
        target_entity_id=target,
        relationship_type="uses_method",
        confidence=0.9,
        extraction_version="v1",
        source_chunk_id="chunk:api",
        supporting_chunk_ids=["chunk:api"],
        provenance_type="chunk",
        paper_version_id="paper-version:arxiv:api:v1",
        graph_index_generation_fingerprint="gen-api",
    )


class FakeGraphRepository:
    def __init__(self) -> None:
        self.paper = _node("paper:arxiv:api", "paper", "API Paper")
        self.method = _node("entity:method:api", "method", "API Method")

    def get_entity(self, entity_id: str):
        if entity_id == self.paper.entity_id:
            return self.paper
        if entity_id == self.method.entity_id:
            return self.method
        return None

    def find_entities_by_canonical_name(self, canonical_name: str, *, entity_type=None, limit=20):
        if canonical_name == "Ambiguous":
            return [_node("entity:method:a", "method", "Ambiguous"), _node("entity:dataset:a", "dataset", "Ambiguous")]
        return []

    def get_direct_paths(self, start_entity_id: str, **kwargs):
        return [
            GraphPathRecord(
                nodes=[self.paper, self.method],
                relationships=[_rel("rel-api", self.paper.entity_id, self.method.entity_id)],
            )
        ]

    def get_shared_entity_paths(self, *args, **kwargs):
        return []

    def get_citing_paper_entity_paths(self, *args, **kwargs):
        return []

    def get_entity_paper_entity_paths(self, *args, **kwargs):
        return []

    def get_citation_neighborhood_paths(self, *args, **kwargs):
        return []


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_graph_repository] = lambda: FakeGraphRepository()
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, GRAPH_MAX_DEPTH=3, GRAPH_DEFAULT_LIMIT=20, GRAPH_MAX_LIMIT=100
    )
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_graph_repository, None)
        app.dependency_overrides.pop(get_settings, None)


class TestGraphSearchEndpoint:
    def test_returns_normalized_graph_evidence(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/search/graph",
            json={"operation": "paper_methods", "entity_id": "paper:arxiv:api"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["operation"] == "paper_methods"
        assert body["start_entity"]["entity_id"] == "paper:arxiv:api"
        assert body["results"][0]["entity"]["entity_id"] == "entity:method:api"
        evidence = body["evidence"][0]
        assert evidence["source"] == "neo4j"
        assert evidence["source_store"] == "neo4j"
        assert evidence["score_kind"] == "graph_path_confidence"
        assert evidence["relationship_ids"] == ["rel-api"]
        assert evidence["paper_version_id"] == "paper-version:arxiv:api:v1"
        assert evidence["source_chunk_ids"] == ["chunk:api"]
        assert evidence["metadata"]["relationships"][0]["graph_index_generation_fingerprint"] == "gen-api"
        assert "neo4j.graph" not in str(body)

    def test_missing_entity_returns_404(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/search/graph",
            json={"operation": "paper_methods", "entity_id": "paper:arxiv:missing"},
        )

        assert response.status_code == 404

    def test_ambiguous_entity_returns_409_with_candidates(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/search/graph",
            json={"operation": "papers_for_dataset", "canonical_name": "Ambiguous"},
        )

        assert response.status_code == 409
        assert len(response.json()["candidates"]) == 2

    def test_bad_limit_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/search/graph",
            json={"operation": "paper_methods", "entity_id": "paper:arxiv:api", "limit": 0},
        )

        assert response.status_code == 400
