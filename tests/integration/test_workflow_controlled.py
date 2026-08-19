import json
from typing import Any

from app.core.config import Settings
from app.domain.enums import EvidenceType, RetrievalStrategy
from app.graph.models import GraphNodeInput, GraphRelationshipInput
from app.generation.answer import AnswerContextBuilder, GroundedAnswerGenerator
from app.generation.citations import CitationValidator
from app.retrieval.critic import EvidenceCriticService, RetrievalRefinementPlanner
from app.retrieval.evidence import EvidenceProvenanceBridge
from app.retrieval.graph_search import GraphRetrievalService
from app.retrieval.hybrid import EvidenceFusionService, HybridRetrievalService
from app.retrieval.planning import QueryAnalysisService, RetrievalPlanner
from app.retrieval.vector_search import VectorSearchService
from app.retrieval.workflow import RetrievalWorkflowService, RetrievalWorkflowStatus
from app.storage.qdrant.models import VectorPoint, VectorPointPayload, build_qdrant_point_id
from app.storage.qdrant.qdrant_repository import QdrantVectorRepository


class ControlledEmbeddingProvider:
    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        if "methodology" in lowered or "approach" in lowered or "method" in lowered:
            return [1.0, 0.0, 0.0, 0.0]
        if "dataset" in lowered:
            return [0.0, 1.0, 0.0, 0.0]
        return [0.0, 0.0, 1.0, 0.0]


class FakePlannerLLM:
    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self.responses = responses
        self.last_usage = None
        self.planner_calls = 0
        self.critic_calls = 0
        self.critic_calls_by_query: dict[str, int] = {}

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-controlled-workflow"

    @property
    def provider_version(self) -> str:
        return "1.0"

    @property
    def temperature(self) -> float:
        return 0.0

    def generate_structured(self, *, system_prompt: str, user_prompt: str, response_model):
        if response_model.__name__ == "EvidenceAssessment":
            self.critic_calls += 1
            query = json.loads(user_prompt)["query"]
            self.critic_calls_by_query[query] = self.critic_calls_by_query.get(query, 0) + 1
            if "Refinement needed methodology" in query:
                if self.critic_calls_by_query[query] > 1:
                    return response_model.model_validate(
                        {
                            "sufficient": True,
                            "coverage": "complete",
                            "recommended_refinement_type": "none",
                        }
                    )
                return response_model.model_validate(
                    {
                        "sufficient": False,
                        "coverage": "partial",
                        "missing_information": ["methodology detail"],
                        "recommended_refinement_type": "vector_expansion",
                    }
                )
            if "Unanswerable methodology" in query:
                return response_model.model_validate(
                    {
                        "sufficient": False,
                        "coverage": "partial",
                        "missing_information": ["missing methodology"],
                        "recommended_refinement_type": "vector_expansion",
                    }
                )
            return response_model.model_validate(
                {
                    "sufficient": True,
                    "coverage": "complete",
                    "recommended_refinement_type": "none",
                    "critic_confidence": 0.9,
                }
            )
        if response_model.__name__ == "GeneratedGroundedAnswer":
            return response_model.model_validate(
                {
                    "text": "The controlled evidence supports the answer [E1].",
                    "used_evidence_markers": ["E1"],
                }
            )
        self.planner_calls += 1
        for query, payload in self.responses.items():
            if query in user_prompt:
                return response_model.model_validate({"query": query, **payload})
        raise AssertionError(f"unexpected query prompt: {user_prompt}")


class CountingGraphService:
    def __init__(self, inner: GraphRetrievalService) -> None:
        self.inner = inner
        self.calls = 0

    def search(self, *args, **kwargs):
        self.calls += 1
        return self.inner.search(*args, **kwargs)


def test_controlled_langgraph_workflow_uses_real_qdrant_and_neo4j(
    qdrant_client,
    qdrant_collection_name,
    neo4j_repository,
) -> None:
    vector_repo = QdrantVectorRepository(qdrant_client, qdrant_collection_name)
    vector_repo.ensure_collection(dimension=4, distance="cosine")
    vector_repo.upsert_chunks(
        [
            _point("chunk:wf:a-method", "paper:arxiv:wf-a", [1.0, 0.0, 0.0, 0.0], "Paper A methodology uses Method X."),
            _point("chunk:wf:a-dataset", "paper:arxiv:wf-a", [0.0, 1.0, 0.0, 0.0], "Paper A evaluates on Dataset X."),
            _point("chunk:wf:b-method", "paper:arxiv:wf-b", [0.5, 0.5, 0.0, 0.0], "Paper B also uses Method X."),
            _point("chunk:wf:b-dataset", "paper:arxiv:wf-b", [0.0, 0.95, 0.0, 0.0], "Paper B evaluates on Dataset X."),
            _point("chunk:wf:b-cites-a", "paper:arxiv:wf-b", [0.0, 0.5, 0.5, 0.0], "Paper B cites Paper A."),
            _point(
                "chunk:wf:refine-detail",
                "paper:arxiv:wf-a",
                [0.8, 0.6, 0.0, 0.0],
                "Refinement needed methodology uses a contrastive retrieval encoder.",
            ),
        ]
    )

    neo4j_repository.ensure_schema()
    neo4j_repository.upsert_entities(
        [
            GraphNodeInput(entity_id="paper:arxiv:wf-a", entity_type="paper", canonical_name="Paper A"),
            GraphNodeInput(entity_id="paper:arxiv:wf-b", entity_type="paper", canonical_name="Paper B"),
            GraphNodeInput(entity_id="entity:method:wf-x", entity_type="method", canonical_name="Method X"),
            GraphNodeInput(entity_id="entity:dataset:wf-x", entity_type="dataset", canonical_name="Dataset X"),
            GraphNodeInput(entity_id="entity:task:wf-x", entity_type="task", canonical_name="Task X"),
        ]
    )
    neo4j_repository.upsert_relationships(
        [
            _rel("rel:wf:a-method", "paper:arxiv:wf-a", "entity:method:wf-x", "uses_method", "chunk:wf:a-method"),
            _rel("rel:wf:a-dataset", "paper:arxiv:wf-a", "entity:dataset:wf-x", "evaluated_on", "chunk:wf:a-dataset"),
            _rel("rel:wf:a-task", "paper:arxiv:wf-a", "entity:task:wf-x", "addresses", "chunk:wf:a-method"),
            _rel("rel:wf:b-method", "paper:arxiv:wf-b", "entity:method:wf-x", "uses_method", "chunk:wf:b-method"),
            _rel("rel:wf:b-dataset", "paper:arxiv:wf-b", "entity:dataset:wf-x", "evaluated_on", "chunk:wf:b-dataset"),
            _rel("rel:wf:b-cites-a", "paper:arxiv:wf-b", "paper:arxiv:wf-a", "cites", "chunk:wf:b-cites-a"),
        ]
    )

    responses = {
        "Explain Paper A's methodology.": _analysis("semantic_explanation", "Paper A"),
        "Which datasets does Paper A evaluate on?": _analysis("paper_datasets", "Paper A"),
        "Which papers use the same method as Paper A?": _analysis("shared_methods", "Paper A"),
        "Which datasets are used by papers citing Paper A?": _analysis("datasets_from_citing_papers", "Paper A"),
        "Explain Paper A's approach and list its datasets.": _analysis("mixed_semantic_structural", "Paper A"),
        "Refinement needed methodology for Paper A.": _analysis("semantic_explanation", "Paper A"),
        "Unanswerable methodology for Paper A.": _analysis("semantic_explanation", "Paper A"),
    }
    settings = Settings(
        _env_file=None,
        VECTOR_SEARCH_DEFAULT_TOP_K=1,
        VECTOR_SEARCH_MAX_TOP_K=10,
        GRAPH_DEFAULT_LIMIT=5,
        GRAPH_MAX_LIMIT=10,
        GRAPH_MAX_DEPTH=3,
        HYBRID_DEFAULT_TOP_K=5,
        HYBRID_MAX_TOP_K=10,
        HYBRID_RRF_K=60,
    )
    graph_service = CountingGraphService(
        GraphRetrievalService(neo4j_repository, max_depth=3, default_limit=5, max_limit=10)
    )
    fake_llm = FakePlannerLLM(responses)
    workflow = RetrievalWorkflowService(
        analysis_service=QueryAnalysisService(fake_llm, settings=settings),
        planner=RetrievalPlanner(neo4j_repository, settings=settings),
        retrieval_service=HybridRetrievalService(
            vector_service=VectorSearchService(
                    ControlledEmbeddingProvider(), vector_repo, default_top_k=1, max_top_k=10
            ),
            graph_service=graph_service,
            provenance_bridge=EvidenceProvenanceBridge(
                vector_repo,
                max_supporting_chunks=5,
                expected_vector_generation_fingerprint="vector-current",
            ),
            fusion_service=EvidenceFusionService(rrf_k=60),
            default_top_k=5,
            max_top_k=10,
        ),
        critic_service=EvidenceCriticService(fake_llm, settings=settings),
        refinement_planner=RetrievalRefinementPlanner(settings=settings),
        settings=settings,
        answer_context_builder=AnswerContextBuilder(settings=settings),
        answer_generator=GroundedAnswerGenerator(fake_llm, settings=settings),
        citation_validator=CitationValidator(settings=settings),
        enable_answer_generation=True,
    )

    semantic = workflow.run("Explain Paper A's methodology.")
    semantic_refined = workflow.run("Refinement needed methodology for Paper A.")
    structural = workflow.run("Which datasets does Paper A evaluate on?")
    shared = workflow.run("Which papers use the same method as Paper A?")
    multi_hop = workflow.run("Which datasets are used by papers citing Paper A?")
    mixed = workflow.run("Explain Paper A's approach and list its datasets.")
    max_round = workflow.run("Unanswerable methodology for Paper A.")

    assert semantic.status == RetrievalWorkflowStatus.SUCCESS
    assert semantic.generated_answer is not None
    assert semantic.citation_validation.validation_status.value == "valid"
    assert semantic.retrieval_plan.strategy == RetrievalStrategy.VECTOR
    assert any(item.chunk_id == "chunk:wf:a-method" for item in semantic.evidence)
    assert semantic.retrieval_round == 1

    assert semantic_refined.status == RetrievalWorkflowStatus.SUCCESS
    assert semantic_refined.generated_answer is not None
    assert semantic_refined.citations
    assert semantic_refined.retrieval_round == 2
    assert semantic_refined.refinement is not None
    assert semantic_refined.refinement.refinement_type.value == "vector_expansion"
    assert any(item.chunk_id == "chunk:wf:refine-detail" for item in semantic_refined.evidence)

    assert structural.status == RetrievalWorkflowStatus.SUCCESS
    assert structural.generated_answer is not None
    assert structural.citations[0].relationship_ids
    assert structural.retrieval_plan.strategy == RetrievalStrategy.GRAPH
    assert structural.retrieval_plan.graph_operation == "paper_datasets"
    assert any("rel:wf:a-dataset" in item.relationship_ids for item in structural.evidence)

    assert shared.status == RetrievalWorkflowStatus.SUCCESS
    assert shared.generated_answer is not None
    assert shared.retrieval_plan.strategy == RetrievalStrategy.GRAPH
    assert shared.retrieval_plan.graph_operation == "shared_methods"
    assert any("paper:arxiv:wf-b" in item.entity_ids for item in shared.evidence)

    assert multi_hop.status == RetrievalWorkflowStatus.SUCCESS
    assert multi_hop.generated_answer is not None
    assert multi_hop.retrieval_plan.strategy == RetrievalStrategy.GRAPH
    assert multi_hop.retrieval_plan.graph_operation == "datasets_from_citing_papers"
    assert any("entity:dataset:wf-x" in item.entity_ids for item in multi_hop.evidence)

    assert mixed.status == RetrievalWorkflowStatus.SUCCESS
    assert mixed.generated_answer is not None
    assert mixed.retrieval_plan.strategy == RetrievalStrategy.HYBRID
    assert {item.evidence_type for item in mixed.evidence} >= {
        EvidenceType.TEXT,
        EvidenceType.GRAPH_RELATIONSHIP,
    }

    assert max_round.status == RetrievalWorkflowStatus.INSUFFICIENT_EVIDENCE
    assert max_round.generated_answer is None
    assert max_round.retrieval_round == 2
    assert graph_service.calls == 4


def _analysis(intent: str, paper: str) -> dict[str, Any]:
    return {
        "intent": intent,
        "semantic_retrieval_required": intent in {"semantic_explanation", "mixed_semantic_structural"},
        "structural_retrieval_required": intent != "semantic_explanation",
        "entities": [{"text": paper, "entity_type": "paper"}],
        "planning_confidence": 0.9,
    }


def _payload(chunk_id: str, paper_id: str, text: str) -> VectorPointPayload:
    return VectorPointPayload(
        chunk_id=chunk_id,
        paper_id=paper_id,
        paper_version_id=f"{paper_id}:v1",
        section_id=f"section:{chunk_id}",
        section_type="methodology",
        section_title="Controlled",
        chunk_index=0,
        page_start=1,
        page_end=1,
        source="controlled",
        source_id=paper_id,
        published_year=2026,
        categories=["cs.IR"],
        chunking_version="chunk-v1",
        chunk_config_fingerprint="chunk-fp",
        embedding_provider="controlled_fake",
        embedding_model="workflow-4d-v1",
        embedding_config_fingerprint="workflow-fp",
        vector_generation_fingerprint="vector-current",
        text=text,
    )


def _point(chunk_id: str, paper_id: str, vector: list[float], text: str) -> VectorPoint:
    return VectorPoint(
        point_id=build_qdrant_point_id(chunk_id),
        vector=vector,
        payload=_payload(chunk_id, paper_id, text),
    )


def _rel(
    relationship_id: str,
    source_entity_id: str,
    target_entity_id: str,
    relationship_type: str,
    source_chunk_id: str,
) -> GraphRelationshipInput:
    return GraphRelationshipInput(
        relationship_id=relationship_id,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relationship_type=relationship_type,
        confidence=0.9,
        extraction_version="extract-v1",
        source_chunk_id=source_chunk_id,
        supporting_chunk_ids=[source_chunk_id],
        provenance_type="chunk",
        paper_version_id=f"{source_entity_id}:v1",
        graph_index_generation_fingerprint="graph-current",
    )
