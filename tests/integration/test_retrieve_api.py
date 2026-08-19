from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.embeddings.provider import get_embedding_provider
from app.graph.models import GraphNodeRecord, GraphPathRecord, GraphRelationshipRecord
from app.graph.neo4j_repository import get_graph_repository
from app.main import app
from app.storage.qdrant.models import VectorChunkRecord, VectorSearchHit
from app.storage.qdrant.qdrant_repository import get_vector_repository


def _hit(chunk_id: str) -> VectorSearchHit:
    return VectorSearchHit(
        chunk_id=chunk_id,
        paper_id="paper:arxiv:api",
        paper_version_id="paper-version:arxiv:api:v1",
        section_id="section:api",
        section_type="methodology",
        section_title="Method",
        chunk_index=0,
        page_start=2,
        page_end=3,
        text=f"Text for {chunk_id}",
        vector_generation_fingerprint="vector-api",
        similarity_score=0.8,
    )


def _chunk(chunk_id: str) -> VectorChunkRecord:
    hit = _hit(chunk_id)
    return VectorChunkRecord(
        chunk_id=hit.chunk_id,
        paper_id=hit.paper_id,
        paper_version_id=hit.paper_version_id,
        section_id=hit.section_id,
        section_type=hit.section_type,
        section_title=hit.section_title,
        chunk_index=hit.chunk_index,
        page_start=hit.page_start,
        page_end=hit.page_end,
        text=hit.text,
        vector_generation_fingerprint=hit.vector_generation_fingerprint,
    )


class FakeEmbeddingProvider:
    dimension = 4

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


class FakeVectorRepository:
    def __init__(self) -> None:
        self.search_calls = 0
        self.lookup_calls = 0

    def search(self, *args, **kwargs):
        self.search_calls += 1
        return [_hit("chunk:api-support")]

    def get_by_chunk_ids(self, chunk_ids: list[str]):
        self.lookup_calls += 1
        return [_chunk(chunk_id) for chunk_id in chunk_ids if chunk_id == "chunk:api-support"]


def _node(entity_id: str, entity_type: str, name: str) -> GraphNodeRecord:
    return GraphNodeRecord(entity_id=entity_id, entity_type=entity_type, canonical_name=name)


class FakeGraphRepository:
    def __init__(self) -> None:
        self.paper = _node("paper:arxiv:api", "paper", "API Paper")
        self.dataset = _node("entity:dataset:api", "dataset", "API Dataset")
        self.direct_calls = 0

    def get_entity(self, entity_id: str):
        return self.paper if entity_id == self.paper.entity_id else None

    def find_entities_by_canonical_name(self, canonical_name: str, *, entity_type=None, limit=20):
        return []

    def get_direct_paths(self, *args, **kwargs):
        self.direct_calls += 1
        return [
            GraphPathRecord(
                nodes=[self.paper, self.dataset],
                relationships=[
                    GraphRelationshipRecord(
                        relationship_id="rel:api-dataset",
                        source_entity_id=self.paper.entity_id,
                        target_entity_id=self.dataset.entity_id,
                        relationship_type="evaluated_on",
                        confidence=0.9,
                        extraction_version="extract-api",
                        source_chunk_id="chunk:api-support",
                        supporting_chunk_ids=["chunk:api-support"],
                        provenance_type="chunk",
                        paper_version_id="paper-version:arxiv:api:v1",
                        graph_index_generation_fingerprint="graph-api",
                    )
                ],
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
    vector_repo = FakeVectorRepository()
    graph_repo = FakeGraphRepository()
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vector_repository] = lambda: vector_repo
    app.dependency_overrides[get_graph_repository] = lambda: graph_repo
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        GRAPH_MAX_DEPTH=3,
        GRAPH_DEFAULT_LIMIT=20,
        GRAPH_MAX_LIMIT=100,
        EVIDENCE_MAX_SUPPORTING_CHUNKS=5,
        HYBRID_RRF_K=60,
        HYBRID_DEFAULT_TOP_K=10,
        HYBRID_MAX_TOP_K=50,
    )
    try:
        with TestClient(app) as test_client:
            test_client.fake_vector_repo = vector_repo
            test_client.fake_graph_repo = graph_repo
            yield test_client
    finally:
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vector_repository, None)
        app.dependency_overrides.pop(get_graph_repository, None)
        app.dependency_overrides.pop(get_settings, None)


class TestRetrieveEndpoint:
    def test_vector_strategy_does_not_call_graph_retrieval(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/search/retrieve",
            json={"query": "dataset", "strategy": "vector"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["vector_result"]["evidence_count"] == 1
        assert body["graph_result"] is None
        assert client.fake_vector_repo.search_calls == 1
        assert client.fake_graph_repo.direct_calls == 0
        assert body["evidence"][0]["evidence"]["score_kind"] == "vector_similarity"

    def test_graph_strategy_does_not_call_semantic_vector_search(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/search/retrieve",
            json={
                "query": "dataset",
                "strategy": "graph",
                "graph": {"operation": "paper_datasets", "entity_id": "paper:arxiv:api"},
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["vector_result"] is None
        assert body["graph_result"]["evidence_count"] == 1
        assert client.fake_vector_repo.search_calls == 0
        assert client.fake_vector_repo.lookup_calls == 1
        assert {item["evidence"]["evidence_type"] for item in body["evidence"]} == {
            "graph_relationship",
            "text",
        }

    def test_hybrid_strategy_returns_fused_diagnostics_and_pool(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/search/retrieve",
            json={
                "query": "dataset",
                "strategy": "hybrid",
                "graph": {"operation": "paper_datasets", "entity_id": "paper:arxiv:api"},
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["diagnostics"]["fusion_method"] == "rrf"
        assert body["diagnostics"]["vector_candidates"] == 1
        assert body["diagnostics"]["graph_candidates"] == 1
        assert body["diagnostics"]["cross_store_links"] == 1
        assert body["evidence_pool"][0]["pool_id"] == "E1"
        graph_item = next(
            item for item in body["evidence"] if item["evidence"]["evidence_type"] == "graph_relationship"
        )
        assert graph_item["cross_store_supported"] is True

    def test_bad_top_k_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/search/retrieve",
            json={"query": "dataset", "strategy": "vector", "top_k": 0},
        )

        assert response.status_code == 400
