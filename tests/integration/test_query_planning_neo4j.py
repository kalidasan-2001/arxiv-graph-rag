from app.core.config import Settings
from app.domain.enums import EntityType
from app.graph.models import GraphNodeInput
from app.retrieval.planning import (
    QueryAnalysisService,
    QueryIntent,
    QueryPlanningService,
    RetrievalPlanner,
)
from tests.retrieval.test_query_planning import FakePlannerLLM, _analysis


def test_query_planning_resolves_entity_with_real_neo4j(neo4j_repository) -> None:
    neo4j_repository.ensure_schema()
    neo4j_repository.upsert_entities(
        [
            GraphNodeInput(
                entity_id="paper:arxiv:neo",
                entity_type="paper",
                canonical_name="Neo Paper",
                aliases=[],
                properties={"source": "arxiv", "source_id": "neo"},
            )
        ]
    )
    settings = Settings(_env_file=None)
    llm = FakePlannerLLM(
        _analysis(
            "Which datasets does Neo Paper evaluate on?",
            QueryIntent.PAPER_DATASETS,
            entity_text="Neo Paper",
            entity_type=EntityType.PAPER,
        )
    )
    result = QueryPlanningService(
        QueryAnalysisService(llm, settings=settings),
        RetrievalPlanner(neo4j_repository, settings=settings),
        settings=settings,
    ).plan("Which datasets does Neo Paper evaluate on?")

    assert result.status == "ok"
    assert result.plan.strategy == "graph"
    assert result.plan.graph_request["entity_id"] == "paper:arxiv:neo"
