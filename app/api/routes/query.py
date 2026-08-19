"""Query-analysis and retrieval-planning routes."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, field_validator

from app.core.config import Settings, get_settings
from app.domain.ids import normalize_whitespace
from app.embeddings.provider import EmbeddingProvider, get_embedding_provider
from app.graph.neo4j_repository import get_graph_repository
from app.graph.repository import GraphRepository
from app.generation.answer import AnswerContextBuilder, GroundedAnswerGenerator
from app.generation.citations import CitationValidator
from app.llm.provider import LLMProvider, get_llm_provider
from app.retrieval.critic import EvidenceCriticService, RetrievalRefinementPlanner
from app.retrieval.evidence import EvidenceProvenanceBridge
from app.retrieval.graph_search import GraphRetrievalService
from app.retrieval.hybrid import EvidenceFusionService, HybridRetrievalService
from app.retrieval.planning import (
    QueryAnalysisService,
    QueryPlanningResult,
    QueryPlanningService,
    RetrievalPlanner,
)
from app.retrieval.vector_search import VectorSearchService
from app.retrieval.workflow import RetrievalWorkflowResult, RetrievalWorkflowService
from app.storage.qdrant.qdrant_repository import get_vector_repository
from app.storage.qdrant.repository import VectorRepository

router = APIRouter(prefix="/query", tags=["query"])


class QueryPlanRequest(BaseModel):
    """Request body for `POST /api/v1/query/plan`."""

    query: str

    @field_validator("query")
    @classmethod
    def _query_required(cls, value: str) -> str:
        normalized = normalize_whitespace(value)
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


@router.post("/plan", response_model=QueryPlanningResult)
def plan_query(
    request: QueryPlanRequest,
    settings: Settings = Depends(get_settings),
    llm_provider: LLMProvider = Depends(get_llm_provider),
    graph_repository: GraphRepository = Depends(get_graph_repository),
) -> QueryPlanningResult:
    """Return a validated retrieval plan without executing retrieval."""

    analysis_service = QueryAnalysisService(llm_provider, settings=settings)
    planner = RetrievalPlanner(graph_repository, settings=settings)
    service = QueryPlanningService(analysis_service, planner, settings=settings)
    return service.plan(request.query)


@router.post("/retrieve", response_model=RetrievalWorkflowResult)
def retrieve_query(
    request: QueryPlanRequest,
    settings: Settings = Depends(get_settings),
    llm_provider: LLMProvider = Depends(get_llm_provider),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    vector_repository: VectorRepository = Depends(get_vector_repository),
    graph_repository: GraphRepository = Depends(get_graph_repository),
) -> RetrievalWorkflowResult:
    """Plan and execute one bounded retrieval workflow without answering."""

    analysis_service = QueryAnalysisService(llm_provider, settings=settings)
    planner = RetrievalPlanner(graph_repository, settings=settings)
    vector_service = VectorSearchService(
        embedding_provider,
        vector_repository,
        default_top_k=settings.VECTOR_SEARCH_DEFAULT_TOP_K,
        max_top_k=settings.VECTOR_SEARCH_MAX_TOP_K,
    )
    graph_service = GraphRetrievalService(
        graph_repository,
        max_depth=settings.GRAPH_MAX_DEPTH,
        default_limit=settings.GRAPH_DEFAULT_LIMIT,
        max_limit=settings.GRAPH_MAX_LIMIT,
    )
    retrieval_service = HybridRetrievalService(
        vector_service=vector_service,
        graph_service=graph_service,
        provenance_bridge=EvidenceProvenanceBridge(
            vector_repository,
            max_supporting_chunks=settings.EVIDENCE_MAX_SUPPORTING_CHUNKS,
        ),
        fusion_service=EvidenceFusionService(rrf_k=settings.HYBRID_RRF_K),
        default_top_k=settings.HYBRID_DEFAULT_TOP_K,
        max_top_k=settings.HYBRID_MAX_TOP_K,
    )
    workflow = RetrievalWorkflowService(
        analysis_service=analysis_service,
        planner=planner,
        retrieval_service=retrieval_service,
        critic_service=EvidenceCriticService(llm_provider, settings=settings),
        refinement_planner=RetrievalRefinementPlanner(settings=settings),
        settings=settings,
    )
    return workflow.run(request.query)


@router.post("/answer", response_model=RetrievalWorkflowResult)
def answer_query(
    request: QueryPlanRequest,
    settings: Settings = Depends(get_settings),
    llm_provider: LLMProvider = Depends(get_llm_provider),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    vector_repository: VectorRepository = Depends(get_vector_repository),
    graph_repository: GraphRepository = Depends(get_graph_repository),
) -> RetrievalWorkflowResult:
    """Run bounded retrieval and generate a grounded answer from the closed evidence pool."""

    analysis_service = QueryAnalysisService(llm_provider, settings=settings)
    planner = RetrievalPlanner(graph_repository, settings=settings)
    vector_service = VectorSearchService(
        embedding_provider,
        vector_repository,
        default_top_k=settings.VECTOR_SEARCH_DEFAULT_TOP_K,
        max_top_k=settings.VECTOR_SEARCH_MAX_TOP_K,
    )
    graph_service = GraphRetrievalService(
        graph_repository,
        max_depth=settings.GRAPH_MAX_DEPTH,
        default_limit=settings.GRAPH_DEFAULT_LIMIT,
        max_limit=settings.GRAPH_MAX_LIMIT,
    )
    retrieval_service = HybridRetrievalService(
        vector_service=vector_service,
        graph_service=graph_service,
        provenance_bridge=EvidenceProvenanceBridge(
            vector_repository,
            max_supporting_chunks=settings.EVIDENCE_MAX_SUPPORTING_CHUNKS,
        ),
        fusion_service=EvidenceFusionService(rrf_k=settings.HYBRID_RRF_K),
        default_top_k=settings.HYBRID_DEFAULT_TOP_K,
        max_top_k=settings.HYBRID_MAX_TOP_K,
    )
    workflow = RetrievalWorkflowService(
        analysis_service=analysis_service,
        planner=planner,
        retrieval_service=retrieval_service,
        critic_service=EvidenceCriticService(llm_provider, settings=settings),
        refinement_planner=RetrievalRefinementPlanner(settings=settings),
        settings=settings,
        answer_context_builder=AnswerContextBuilder(settings=settings),
        answer_generator=GroundedAnswerGenerator(llm_provider, settings=settings),
        citation_validator=CitationValidator(settings=settings),
        enable_answer_generation=True,
    )
    return workflow.run(request.query)
