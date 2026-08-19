"""Search routes for independent retrieval strategies.

Kept thin per CLAUDE.md #28 -- no Qdrant/Neo4j logic, no embedding calls
or Cypher here, just request validation, service invocation, and response
mapping.
"""

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.domain.evidence import EvidenceItem
from app.domain.enums import EntityType, RetrievalStrategy, SectionType
from app.embeddings.provider import EmbeddingProvider, get_embedding_provider
from app.graph.models import GraphNodeRecord, GraphPathRecord
from app.graph.neo4j_repository import get_graph_repository
from app.graph.repository import GraphRepository
from app.retrieval.evidence import EvidenceProvenanceBridge, VectorEvidenceAdapter
from app.retrieval.graph_search import (
    GraphRetrievalService,
    GraphSearchOperation,
    GraphSearchResponseData,
)
from app.retrieval.hybrid import EvidenceFusionService, HybridRetrievalResult, HybridRetrievalService
from app.retrieval.vector_search import VectorSearchService
from app.storage.qdrant.qdrant_repository import get_vector_repository
from app.storage.qdrant.repository import VectorRepository

router = APIRouter(prefix="/search", tags=["search"])


class VectorSearchRequest(BaseModel):
    """Request body for `POST /api/v1/search/vector`."""

    query: str
    top_k: int | None = None
    paper_id: str | None = None
    paper_version_id: str | None = None
    section_type: SectionType | None = None


class VectorSearchResultResponse(BaseModel):
    """One ranked chunk. `similarity_score` is exactly what the configured
    Qdrant distance metric (cosine) reports -- not a probability or a
    calibrated confidence (prompt #40)."""

    chunk_id: str
    paper_id: str
    paper_version_id: str
    section_id: str
    section_type: str
    section_title: str | None
    chunk_index: int
    page_start: int | None
    page_end: int | None
    text: str
    similarity_score: float


class UnifiedEvidenceResponse(BaseModel):
    evidence_id: str
    evidence_type: str
    paper_id: str | None
    paper_version_id: str | None
    chunk_id: str | None
    section_id: str | None
    section_type: str | None
    page_start: int | None
    page_end: int | None
    entity_ids: list[str]
    relationship_ids: list[str]
    source_chunk_ids: list[str]
    text: str | None
    score: float | None
    score_kind: str | None
    source: str
    source_store: str | None
    supporting_text_evidence_ids: list[str]
    provenance: dict[str, Any] | None
    metadata: dict[str, Any]


class VectorSearchResponse(BaseModel):
    """Response body for `POST /api/v1/search/vector`.

    This is deterministic semantic retrieval only -- no LLM is involved,
    and results are never re-ranked by an arbitrary relevance threshold
    (prompt #41): ranked results are returned as-is.
    """

    query: str
    count: int
    results: list[VectorSearchResultResponse]
    evidence: list[UnifiedEvidenceResponse]


class GraphSearchRequest(BaseModel):
    """Request body for `POST /api/v1/search/graph`."""

    operation: GraphSearchOperation
    entity_id: str | None = None
    entity_type: EntityType | None = None
    canonical_name: str | None = None
    depth: int | None = None
    limit: int | None = None


class GraphSearchResultResponse(BaseModel):
    entity: GraphNodeRecord | None
    path: GraphPathRecord
    evidence_id: str
    path_confidence: float
    summary: str


class GraphSearchResponse(BaseModel):
    operation: GraphSearchOperation
    start_entity: GraphNodeRecord
    results: list[GraphSearchResultResponse]
    evidence: list[UnifiedEvidenceResponse]


class RetrieveGraphRequest(BaseModel):
    operation: GraphSearchOperation
    entity_id: str | None = None
    entity_type: EntityType | None = None
    canonical_name: str | None = None
    depth: int | None = None
    limit: int | None = None


class RetrieveRequest(BaseModel):
    """Request body for explicit-strategy retrieval orchestration."""

    query: str
    strategy: RetrievalStrategy
    graph: RetrieveGraphRequest | None = None
    vector_top_k: int | None = None
    graph_limit: int | None = None
    top_k: int | None = None
    paper_id: str | None = None
    paper_version_id: str | None = None
    section_type: SectionType | None = None


class BranchResultResponse(BaseModel):
    strategy: str
    evidence_count: int
    duration_ms: int
    warnings: list[str]


class FusedEvidenceResponse(BaseModel):
    evidence: UnifiedEvidenceResponse
    fusion_score: float
    branch_ranks: dict[str, int]
    branches: list[str]
    cross_store_supported: bool


class EvidencePoolItemResponse(BaseModel):
    pool_id: str
    evidence_id: str


class RetrieveResponse(BaseModel):
    query: str
    strategy: RetrievalStrategy
    evidence: list[FusedEvidenceResponse]
    evidence_pool: list[EvidencePoolItemResponse]
    vector_result: BranchResultResponse | None
    graph_result: BranchResultResponse | None
    warnings: list[str]
    diagnostics: dict[str, Any]


def _evidence_response(item: EvidenceItem) -> UnifiedEvidenceResponse:
    return UnifiedEvidenceResponse(
        evidence_id=item.evidence_id,
        evidence_type=item.evidence_type.value,
        paper_id=item.paper_id,
        paper_version_id=item.paper_version_id,
        chunk_id=item.chunk_id,
        section_id=item.section_id,
        section_type=item.section_type,
        page_start=item.page_start,
        page_end=item.page_end,
        entity_ids=item.entity_ids,
        relationship_ids=item.relationship_ids,
        source_chunk_ids=item.source_chunk_ids,
        text=item.text,
        score=item.score,
        score_kind=item.score_kind.value if item.score_kind else None,
        source=item.source,
        source_store=item.source_store.value if item.source_store else None,
        supporting_text_evidence_ids=item.supporting_text_evidence_ids,
        provenance=item.provenance.model_dump(mode="json") if item.provenance else None,
        metadata=item.metadata,
    )


def _branch_response(result) -> BranchResultResponse | None:
    if result is None:
        return None
    return BranchResultResponse(
        strategy=result.strategy.value,
        evidence_count=len(result.evidence),
        duration_ms=result.duration_ms,
        warnings=result.warnings,
    )


@router.post("/vector", response_model=VectorSearchResponse)
def search_vector(
    request: VectorSearchRequest,
    settings: Settings = Depends(get_settings),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    vector_repository: VectorRepository = Depends(get_vector_repository),
) -> VectorSearchResponse:
    """Embed `query` and return the most semantically similar indexed chunks.

    Not RAG (prompt #36) -- no LLM is called, and no answer is synthesized.
    References (`section_type=references`) are included unless explicitly
    filtered out (prompt #42) -- down-ranking/exclusion is a future
    retrieval/evaluation decision, not made here. Purely a Qdrant + embedding
    operation -- deliberately does not depend on PostgreSQL at all.
    """

    service = VectorSearchService(
        embedding_provider,
        vector_repository,
        default_top_k=settings.VECTOR_SEARCH_DEFAULT_TOP_K,
        max_top_k=settings.VECTOR_SEARCH_MAX_TOP_K,
    )
    results = service.search(
        request.query,
        top_k=request.top_k,
        paper_id=request.paper_id,
        paper_version_id=request.paper_version_id,
        section_type=request.section_type.value if request.section_type else None,
    )
    evidence = [VectorEvidenceAdapter().from_hit(hit) for hit in results]

    return VectorSearchResponse(
        query=request.query,
        count=len(results),
        results=[
            VectorSearchResultResponse(
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
                similarity_score=hit.similarity_score,
            )
            for hit in results
        ],
        evidence=[_evidence_response(item) for item in evidence],
    )


@router.post("/graph", response_model=GraphSearchResponse)
def search_graph(
    request: GraphSearchRequest,
    settings: Settings = Depends(get_settings),
    graph_repository: GraphRepository = Depends(get_graph_repository),
) -> GraphSearchResponse:
    """Run one explicit bounded Neo4j graph retrieval primitive.

    No LLM, no Qdrant, no hybrid fusion, and no caller-supplied Cypher.
    """

    service = GraphRetrievalService(
        graph_repository,
        max_depth=settings.GRAPH_MAX_DEPTH,
        default_limit=settings.GRAPH_DEFAULT_LIMIT,
        max_limit=settings.GRAPH_MAX_LIMIT,
    )
    result: GraphSearchResponseData = service.search(
        operation=request.operation,
        entity_id=request.entity_id,
        entity_type=request.entity_type,
        canonical_name=request.canonical_name,
        depth=request.depth,
        limit=request.limit,
    )
    return GraphSearchResponse(
        operation=result.operation,
        start_entity=result.start_entity,
        results=[
            GraphSearchResultResponse(
                entity=item.entity,
                path=item.path,
                evidence_id=item.evidence_id,
                path_confidence=item.path_confidence,
                summary=item.summary,
            )
            for item in result.results
        ],
        evidence=[
            _evidence_response(item) for item in result.evidence
        ],
    )


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    request: RetrieveRequest,
    settings: Settings = Depends(get_settings),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    vector_repository: VectorRepository = Depends(get_vector_repository),
    graph_repository: GraphRepository = Depends(get_graph_repository),
) -> RetrieveResponse:
    """Run explicit vector, graph, or hybrid retrieval and return fused evidence.

    This is still deterministic retrieval only: no LLM planning, no answer
    generation, and no automatic graph-operation selection.
    """

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
    service = HybridRetrievalService(
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
    graph = request.graph
    result: HybridRetrievalResult = service.retrieve(
        query=request.query,
        strategy=request.strategy,
        top_k=request.top_k,
        vector_top_k=request.vector_top_k,
        graph_operation=graph.operation if graph else None,
        entity_id=graph.entity_id if graph else None,
        entity_type=graph.entity_type if graph else None,
        canonical_name=graph.canonical_name if graph else None,
        graph_depth=graph.depth if graph else None,
        graph_limit=request.graph_limit if request.graph_limit is not None else (graph.limit if graph else None),
        paper_id=request.paper_id,
        paper_version_id=request.paper_version_id,
        section_type=request.section_type.value if request.section_type else None,
    )
    return RetrieveResponse(
        query=result.query,
        strategy=result.strategy,
        evidence=[
            FusedEvidenceResponse(
                evidence=_evidence_response(item.evidence),
                fusion_score=item.fusion_score,
                branch_ranks=item.branch_ranks,
                branches=item.branches,
                cross_store_supported=item.cross_store_supported,
            )
            for item in result.evidence
        ],
        evidence_pool=[
            EvidencePoolItemResponse(pool_id=item.pool_id, evidence_id=item.evidence.evidence_id)
            for item in result.evidence_pool.items
        ],
        vector_result=_branch_response(result.vector_result),
        graph_result=_branch_response(result.graph_result),
        warnings=result.warnings,
        diagnostics=result.diagnostics,
    )
