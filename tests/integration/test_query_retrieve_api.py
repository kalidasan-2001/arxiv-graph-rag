from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.domain.enums import EntityType
from app.embeddings.provider import get_embedding_provider
from app.graph.neo4j_repository import get_graph_repository
from app.llm.provider import get_llm_provider
from app.main import app
from app.storage.qdrant.qdrant_repository import get_vector_repository
from tests.integration.test_retrieve_api import (
    FakeEmbeddingProvider,
    FakeGraphRepository,
    FakeVectorRepository,
)


class FakePlannerLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.critic_calls = 0
        self.answer_calls = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-workflow-planner"

    @property
    def provider_version(self) -> str:
        return "1.0"

    @property
    def temperature(self) -> float:
        return 0.0

    def generate_structured(self, *, system_prompt: str, user_prompt: str, response_model, max_output_tokens=None):
        self.calls += 1
        if response_model.__name__ == "EvidenceAssessment":
            self.critic_calls += 1
            return response_model.model_validate(
                {
                    "sufficient": True,
                    "coverage": "complete",
                    "recommended_refinement_type": "none",
                    "critic_confidence": 0.9,
                }
            )
        if response_model.__name__ == "GeneratedGroundedAnswer":
            self.answer_calls += 1
            return response_model.model_validate(
                {
                    "text": "API Paper evaluates on API Dataset [E1].",
                    "used_evidence_markers": ["E1"],
                    "citations": ["E999"],
                }
            )
        payload: dict[str, Any] = {
            "query": "Which datasets does API Paper evaluate on?",
            "intent": "paper_datasets",
            "semantic_retrieval_required": False,
            "structural_retrieval_required": True,
            "entities": [{"text": "API Paper", "entity_type": EntityType.PAPER}],
            "planning_confidence": 0.9,
        }
        return response_model.model_validate(payload)


@pytest.fixture
def client() -> Iterator[TestClient]:
    vector_repo = FakeVectorRepository()
    graph_repo = FakeGraphRepository()
    graph_repo.find_entities_by_canonical_name = lambda canonical_name, *, entity_type=None, limit=20: [
        graph_repo.paper
    ] if canonical_name == "API Paper" and entity_type == "paper" else []
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    app.dependency_overrides[get_llm_provider] = lambda: FakePlannerLLM()
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_vector_repository] = lambda: vector_repo
    app.dependency_overrides[get_graph_repository] = lambda: graph_repo
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_settings, None)
        app.dependency_overrides.pop(get_llm_provider, None)
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vector_repository, None)
        app.dependency_overrides.pop(get_graph_repository, None)


def test_query_retrieve_endpoint_runs_langgraph_workflow(client: TestClient) -> None:
    response = client.post(
        "/api/v1/query/retrieve",
        json={"query": "Which datasets does API Paper evaluate on?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUCCESS"
    assert body["analysis"]["intent"] == "paper_datasets"
    assert body["retrieval_plan"]["strategy"] == "graph"
    assert body["retrieval_plan"]["graph_operation"] == "paper_datasets"
    assert body["evidence_pool"]["items"][0]["pool_id"] == "E1"
    assert [event["node"] for event in body["trace"]] == [
        "analyze_query",
        "resolve_entities",
        "build_plan",
        "execute_retrieval",
        "build_evidence_pool",
        "evaluate_evidence",
    ]
    assert body["evidence_assessment"]["sufficient"] is True
    assert body["retrieval_round"] == 1
    assert body["refinement"] is None
    assert body["evidence_history"][0]["retrieval_round"] == 1
    assert body["generated_answer"] is None
    assert body["answer"] is None
    assert body["citations"] == []
    assert body["citation_validation"] is None


def test_query_answer_endpoint_generates_grounded_answer(client: TestClient) -> None:
    response = client.post(
        "/api/v1/query/answer",
        json={"query": "Which datasets does API Paper evaluate on?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUCCESS"
    assert body["generated_answer"]["text"] == "API Paper evaluates on API Dataset [E1]."
    assert body["answer"] == "API Paper evaluates on API Dataset [1]."
    assert body["validated_answer"]["raw_text"] == "API Paper evaluates on API Dataset [E1]."
    assert body["citation_validation"]["validation_status"] == "valid"
    assert body["citations"][0]["evidence_label"] == "E1"
    assert body["citations"][0]["relationship_ids"] == ["rel:api-dataset"]
    assert body["generated_answer"]["used_evidence_markers"] == ["E1"]
    assert body["answer_context"]["evidence_items"][0]["pool_id"] == "E1"
    assert body["answer_generation_metadata"]["citation_validation"] == "valid"
    assert [event["node"] for event in body["trace"]] == [
        "analyze_query",
        "resolve_entities",
        "build_plan",
        "execute_retrieval",
        "build_evidence_pool",
        "evaluate_evidence",
        "prepare_answer_context",
        "generate_answer",
        "validate_citations",
        "finalize_answer",
    ]
    assert body["final_status"] == "answered"
    assert body["confidence"] == "high"
    assert body["final_answer"]["answer"] == "API Paper evaluates on API Dataset [1]."
    assert body["grounding"]["allow_answer"] is True
    assert body["grounding"]["diagnostics"]["trusted_citation_count"] == 1
