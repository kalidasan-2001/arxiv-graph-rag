from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.domain.enums import EntityType
from app.graph.models import GraphNodeRecord
from app.graph.neo4j_repository import get_graph_repository
from app.llm.provider import get_llm_provider
from app.main import app
from app.retrieval.planning import QueryIntent


def _node(entity_id: str, entity_type: str, name: str) -> GraphNodeRecord:
    return GraphNodeRecord(entity_id=entity_id, entity_type=entity_type, canonical_name=name)


class FakeGraphRepository:
    def __init__(self) -> None:
        self.paper = _node("paper:arxiv:api", "paper", "API Paper")

    def get_entity(self, entity_id: str):
        return self.paper if entity_id == self.paper.entity_id else None

    def find_entities_by_canonical_name(self, canonical_name: str, *, entity_type=None, limit=20):
        if canonical_name == "API Paper" and entity_type == "paper":
            return [self.paper]
        return []


class FakePlannerLLM:
    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-planner"

    @property
    def provider_version(self) -> str:
        return "1.0"

    @property
    def temperature(self) -> float:
        return 0.0

    def generate_structured(self, *, system_prompt: str, user_prompt: str, response_model):
        payload: dict[str, Any] = {
            "query": "Which datasets does API Paper evaluate on?",
            "intent": QueryIntent.PAPER_DATASETS,
            "semantic_retrieval_required": False,
            "structural_retrieval_required": True,
            "entities": [{"text": "API Paper", "entity_type": EntityType.PAPER}],
            "planning_confidence": 0.9,
        }
        return response_model.model_validate(payload)


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    app.dependency_overrides[get_llm_provider] = lambda: FakePlannerLLM()
    app.dependency_overrides[get_graph_repository] = lambda: FakeGraphRepository()
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_settings, None)
        app.dependency_overrides.pop(get_llm_provider, None)
        app.dependency_overrides.pop(get_graph_repository, None)


def test_query_plan_endpoint_returns_validated_plan(client: TestClient) -> None:
    response = client.post(
        "/api/v1/query/plan",
        json={"query": "Which datasets does API Paper evaluate on?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["analysis"]["intent"] == "paper_datasets"
    assert body["plan"]["strategy"] == "graph"
    assert body["plan"]["graph_operation"] == "paper_datasets"
    assert body["plan"]["graph_request"]["entity_id"] == "paper:arxiv:api"
    assert body["diagnostics"]["planner_provider"] == "fake"
