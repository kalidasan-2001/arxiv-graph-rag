"""Paper discovery routes.

Kept thin per CLAUDE.md #28: request validation, dependency resolution,
service invocation, response mapping -- no arXiv calls, no SQL, no XML
parsing, and no ORM objects here (architecture sanity check, prompt #46).
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Body, Depends, Query
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    ChunkArtifactNotFoundError,
    GraphExtractionArtifactNotFoundError,
    InvalidSearchQueryError,
    ParseArtifactNotFoundError,
)
from app.domain.enums import IngestionStatus, SectionType
from app.embeddings.provider import EmbeddingProvider, get_embedding_provider
from app.ingestion.chunking.section_chunker import SectionAwareChunker
from app.ingestion.chunking.service import ChunkingService
from app.ingestion.chunking.storage import ChunkArtifactStorage
from app.ingestion.discovery.arxiv_client import ArxivClient, get_arxiv_client
from app.ingestion.discovery.models import PaperSearchQuery, SortBy, SortOrder
from app.ingestion.discovery.service import PaperDiscoveryService
from app.ingestion.download.client import PdfDownloadClient, get_pdf_download_client
from app.ingestion.download.service import PdfAcquisitionService
from app.ingestion.download.storage import PaperStorage
from app.ingestion.graph_extraction.service import ScientificKnowledgeExtractionService
from app.ingestion.graph_extraction.storage import GraphExtractionArtifactStorage
from app.ingestion.graph_indexing.service import GraphIndexingService
from app.ingestion.paper_resolution import resolve_paper, resolve_version
from app.ingestion.parsing.pymupdf_parser import PyMuPDFParser
from app.ingestion.parsing.service import PaperParsingService
from app.ingestion.parsing.storage import ParsedArtifactStorage
from app.ingestion.vector_indexing.service import VectorIndexingService
from app.llm.provider import LLMProvider, get_llm_provider
from app.storage.postgres.repositories.ingestion import IngestionRepository
from app.storage.postgres.repositories.papers import PaperRepository
from app.storage.postgres.session import get_db_session
from app.storage.qdrant.qdrant_repository import get_vector_repository
from app.storage.qdrant.repository import VectorRepository
from app.graph.neo4j_repository import get_graph_repository
from app.graph.repository import GraphRepository

router = APIRouter(prefix="/papers", tags=["papers"])


class PaperSearchResultResponse(BaseModel):
    """One search result. Deliberately not the domain `Paper` model itself
    (CLAUDE.md #27: API schemas stay separate from domain/DB models)."""

    paper_id: str
    source_id: str
    latest_version: str | None
    title: str
    authors: list[str]
    abstract: str | None
    categories: list[str]
    published_at: datetime | None
    updated_at: datetime | None
    pdf_url: str | None
    already_known: bool


class PaperSearchResponse(BaseModel):
    """Response body for `GET /api/v1/papers/search`."""

    query: str
    count: int
    results: list[PaperSearchResultResponse]


@router.get("/search", response_model=PaperSearchResponse)
def search_papers(
    q: str = Query(..., min_length=1, description="Search query, e.g. 'graph rag'"),
    max_results: int | None = Query(None, gt=0),
    sort_by: SortBy = Query(SortBy.RELEVANCE),
    sort_order: SortOrder = Query(SortOrder.DESCENDING),
    category: list[str] = Query(default=[], description="Repeatable, e.g. cat.AI"),
    published_after: datetime | None = Query(None),
    published_before: datetime | None = Query(None),
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_db_session),
    arxiv_client: ArxivClient = Depends(get_arxiv_client),
) -> PaperSearchResponse:
    """Search arXiv and persist discovered paper metadata.

    Discovered papers are recorded in PostgreSQL (metadata only) but this
    endpoint never downloads a PDF and never creates an ingestion job --
    a discovered paper is not yet an ingested, query-ready one.
    """

    try:
        query = PaperSearchQuery(
            query=q,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
            categories=category,
            published_after=published_after,
            published_before=published_before,
        )
    except ValidationError as exc:
        raise InvalidSearchQueryError(str(exc)) from exc

    service = PaperDiscoveryService(settings, arxiv_client, PaperRepository(session))
    results = service.search(query)

    return PaperSearchResponse(
        query=query.query,
        count=len(results),
        results=[
            PaperSearchResultResponse(
                paper_id=result.paper.paper_id,
                source_id=result.paper.source_id,
                latest_version=result.latest_version,
                title=result.paper.title,
                authors=result.paper.authors,
                abstract=result.paper.abstract,
                categories=result.paper.categories,
                published_at=result.paper.published_at,
                updated_at=result.paper.updated_at,
                pdf_url=result.paper.pdf_url,
                already_known=result.already_known,
            )
            for result in results
        ],
    )


class IngestRequest(BaseModel):
    """Request body for `POST /api/v1/papers/{paper_id}/ingest`.

    `paper_version_id` is optional: when omitted, the documented policy
    (prompt #4) is to use the latest discovered version for the logical
    paper -- never to silently invent one.
    """

    paper_version_id: str | None = None


class IngestResponse(BaseModel):
    """Response body for `POST /api/v1/papers/{paper_id}/ingest`.

    No filesystem paths or other storage implementation details are
    exposed (prompt #5) -- only the operational facts a caller needs.
    """

    ingestion_job_id: str
    paper_id: str
    paper_version_id: str
    status: str
    artifact_available: bool
    artifact_reused: bool


@router.post("/{paper_id}/ingest", response_model=IngestResponse)
def ingest_paper(
    paper_id: str,
    request: IngestRequest = Body(default_factory=IngestRequest),
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_db_session),
    pdf_client: PdfDownloadClient = Depends(get_pdf_download_client),
) -> IngestResponse:
    """Explicitly start (or safely reuse) PDF acquisition for a discovered paper.

    Searching arXiv never triggers this -- ingestion only ever happens
    here, on an explicit request (prompt #3: `search != ingest`). Stops at
    a validated, checksummed, stored PDF; never parses it.
    """

    storage = PaperStorage(settings)
    service = PdfAcquisitionService(
        settings,
        pdf_client,
        storage,
        PaperRepository(session),
        IngestionRepository(session),
    )
    result = service.ingest(paper_id, request.paper_version_id)

    return IngestResponse(
        ingestion_job_id=result.job.ingestion_job_id,
        paper_id=result.job.paper_id,
        paper_version_id=result.job.paper_version_id,
        status=result.job.status.value,
        artifact_available=result.job.status == IngestionStatus.DOWNLOADED,
        artifact_reused=result.artifact_reused,
    )


class ParseRequest(BaseModel):
    """Request body for `POST /api/v1/papers/{paper_id}/parse`.

    Same optional-version policy as `IngestRequest`: omitted means "the
    latest discovered version," never a silently invented one.
    """

    paper_version_id: str | None = None


class ParseResponse(BaseModel):
    """Response body for `POST /api/v1/papers/{paper_id}/parse`."""

    ingestion_job_id: str
    paper_id: str
    paper_version_id: str
    status: str
    page_count: int
    section_count: int
    warning_count: int
    parse_reused: bool


@router.post("/{paper_id}/parse", response_model=ParseResponse)
def parse_paper(
    paper_id: str,
    request: ParseRequest = Body(default_factory=ParseRequest),
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_db_session),
) -> ParseResponse:
    """Explicitly parse a downloaded PDF into a structured document.

    Requires the paper version to already be `DOWNLOADED` (prompt #7-8) --
    never triggered automatically by search or ingest, and never continues
    into chunking.
    """

    service = PaperParsingService(
        PyMuPDFParser(),
        PaperStorage(settings),
        ParsedArtifactStorage(settings),
        PaperRepository(session),
        IngestionRepository(session),
    )
    result = service.parse(paper_id, request.paper_version_id)

    return ParseResponse(
        ingestion_job_id=result.job.ingestion_job_id,
        paper_id=result.job.paper_id,
        paper_version_id=result.job.paper_version_id,
        status=result.job.status.value,
        page_count=result.document.page_count,
        section_count=len(result.document.sections),
        warning_count=len(result.document.warnings),
        parse_reused=result.parse_reused,
    )


class DocumentSectionResponse(BaseModel):
    section_id: str
    section_type: str
    title: str | None
    order: int
    page_start: int | None
    page_end: int | None
    text: str | None = None


class DocumentResponse(BaseModel):
    """Response body for `GET /api/v1/papers/{paper_id}/document`."""

    paper_id: str
    paper_version_id: str
    page_count: int
    parser_name: str
    parser_version: str
    warnings: list[str]
    sections: list[DocumentSectionResponse]


@router.get("/{paper_id}/document", response_model=DocumentResponse)
def get_parsed_document(
    paper_id: str,
    paper_version_id: str | None = Query(None),
    include_text: bool = Query(False, description="Include full section text bodies"),
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_db_session),
) -> DocumentResponse:
    """Inspect the structured parse result for a paper version (development/demo).

    Section bodies are omitted by default (`include_text=false`) since a
    full paper's text can be large -- pass `include_text=true` to include it.
    """

    paper_repo = PaperRepository(session)
    paper = resolve_paper(paper_repo, paper_id)
    version = resolve_version(paper_repo, paper, paper_version_id)

    if not version.parsed_artifact_path:
        raise ParseArtifactNotFoundError(
            f"paper version {version.paper_version_id} has not been parsed yet"
        )

    document = ParsedArtifactStorage(settings).try_read(
        source=paper.source, source_id=paper.source_id, version=version.version
    )
    if document is None:
        raise ParseArtifactNotFoundError(
            f"parsed artifact for {version.paper_version_id} is missing or corrupt"
        )

    return DocumentResponse(
        paper_id=document.paper_id,
        paper_version_id=document.paper_version_id,
        page_count=document.page_count,
        parser_name=document.parser_name,
        parser_version=document.parser_version,
        warnings=[warning.value for warning in document.warnings],
        sections=[
            DocumentSectionResponse(
                section_id=section.section_id,
                section_type=section.section_type.value,
                title=section.title,
                order=section.order,
                page_start=section.page_start,
                page_end=section.page_end,
                text=section.text if include_text else None,
            )
            for section in document.sections
        ],
    )


class ChunkRequest(BaseModel):
    """Request body for `POST /api/v1/papers/{paper_id}/chunk`.

    Same optional-version policy as `ParseRequest`/`IngestRequest`.
    """

    paper_version_id: str | None = None


class ChunkResponse(BaseModel):
    """Response body for `POST /api/v1/papers/{paper_id}/chunk`."""

    ingestion_job_id: str
    paper_id: str
    paper_version_id: str
    status: str
    chunk_count: int
    chunking_version: str
    chunk_config_fingerprint: str
    chunk_reused: bool


@router.post("/{paper_id}/chunk", response_model=ChunkResponse)
def chunk_paper(
    paper_id: str,
    request: ChunkRequest = Body(default_factory=ChunkRequest),
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_db_session),
) -> ChunkResponse:
    """Explicitly split a parsed document into section-aware chunks.

    Requires the paper version to already be `PARSED` (prompt #27/#31 --
    chunking always starts from `parsed.json`, never `paper.pdf`
    directly). Never triggered automatically by parsing, and never
    continues into embedding.
    """

    service = ChunkingService(
        SectionAwareChunker(settings),
        ParsedArtifactStorage(settings),
        ChunkArtifactStorage(settings),
        PaperRepository(session),
        IngestionRepository(session),
    )
    result = service.chunk(paper_id, request.paper_version_id)

    return ChunkResponse(
        ingestion_job_id=result.job.ingestion_job_id,
        paper_id=result.job.paper_id,
        paper_version_id=result.job.paper_version_id,
        status=result.job.status.value,
        chunk_count=len(result.document.chunks),
        chunking_version=result.document.chunking.version,
        chunk_config_fingerprint=result.document.chunking.config_fingerprint,
        chunk_reused=result.chunk_reused,
    )


class ChunkPreviewResponse(BaseModel):
    chunk_id: str
    section_type: str
    section_title: str | None
    chunk_index: int
    token_count: int
    page_start: int | None
    page_end: int | None
    text_preview: str
    text: str | None = None


class ChunkListResponse(BaseModel):
    """Response body for `GET /api/v1/papers/{paper_id}/chunks`."""

    paper_id: str
    paper_version_id: str
    total_count: int
    chunks: list[ChunkPreviewResponse]


_TEXT_PREVIEW_LENGTH = 200


@router.get("/{paper_id}/chunks", response_model=ChunkListResponse)
def list_chunks(
    paper_id: str,
    paper_version_id: str | None = Query(None),
    section_type: SectionType | None = Query(None),
    limit: int = Query(50, gt=0, le=500),
    offset: int = Query(0, ge=0),
    include_text: bool = Query(False, description="Include each chunk's full text"),
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_db_session),
) -> ChunkListResponse:
    """Inspect the chunk corpus for a paper version (development/demo).

    Chunks cannot be created or edited via this API (prompt #34) -- they
    are deterministic derived artifacts; rerun `/chunk` with a new
    configuration/version to regenerate them.
    """

    paper_repo = PaperRepository(session)
    paper = resolve_paper(paper_repo, paper_id)
    version = resolve_version(paper_repo, paper, paper_version_id)

    if not version.chunked_artifact_path:
        raise ChunkArtifactNotFoundError(
            f"paper version {version.paper_version_id} has not been chunked yet"
        )

    document = ChunkArtifactStorage(settings).try_read(
        source=paper.source, source_id=paper.source_id, version=version.version
    )
    if document is None:
        raise ChunkArtifactNotFoundError(
            f"chunk artifact for {version.paper_version_id} is missing or corrupt"
        )

    chunks = document.chunks
    if section_type is not None:
        chunks = [chunk for chunk in chunks if chunk.section_type == section_type]
    total_count = len(chunks)
    page = chunks[offset : offset + limit]

    return ChunkListResponse(
        paper_id=document.paper_id,
        paper_version_id=document.paper_version_id,
        total_count=total_count,
        chunks=[
            ChunkPreviewResponse(
                chunk_id=chunk.chunk_id,
                section_type=chunk.section_type.value,
                section_title=chunk.metadata.get("section_title"),
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text_preview=chunk.text[:_TEXT_PREVIEW_LENGTH],
                text=chunk.text if include_text else None,
            )
            for chunk in page
        ],
    )


class VectorIndexRequest(BaseModel):
    """Request body for `POST /api/v1/papers/{paper_id}/vector-index`.

    Same optional-version policy as `ChunkRequest`/`ParseRequest`/`IngestRequest`.
    """

    paper_version_id: str | None = None


class VectorIndexResponse(BaseModel):
    """Response body for `POST /api/v1/papers/{paper_id}/vector-index`."""

    ingestion_job_id: str
    paper_id: str
    paper_version_id: str
    status: str
    vector_count: int
    embedding_provider: str
    embedding_model: str
    embedding_config_fingerprint: str
    vector_generation_fingerprint: str
    artifact_reused: bool


@router.post("/{paper_id}/vector-index", response_model=VectorIndexResponse)
def vector_index_paper(
    paper_id: str,
    request: VectorIndexRequest = Body(default_factory=VectorIndexRequest),
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_db_session),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    vector_repository: VectorRepository = Depends(get_vector_repository),
) -> VectorIndexResponse:
    """Explicitly embed a chunked paper version's chunks and index them
    into Qdrant.

    Requires the paper version to already be `CHUNKED` (prompt #21/#28 --
    vector indexing always starts from `chunks.json`, never re-parses or
    re-chunks). Never triggered automatically by chunking, and never
    continues into graph extraction.
    """

    service = VectorIndexingService(
        embedding_provider,
        vector_repository,
        ChunkArtifactStorage(settings),
        PaperRepository(session),
        IngestionRepository(session),
        collection_name=settings.QDRANT_COLLECTION,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )
    result = service.index_paper_version(paper_id, request.paper_version_id)

    return VectorIndexResponse(
        ingestion_job_id=result.job.ingestion_job_id,
        paper_id=result.job.paper_id,
        paper_version_id=result.job.paper_version_id,
        status=result.job.status.value,
        vector_count=result.vector_count,
        embedding_provider=result.embedding_provider,
        embedding_model=result.embedding_model,
        embedding_config_fingerprint=result.embedding_config_fingerprint,
        vector_generation_fingerprint=result.vector_generation_fingerprint,
        artifact_reused=result.index_reused,
    )


class ExtractGraphRequest(BaseModel):
    """Request body for `POST /api/v1/papers/{paper_id}/extract-graph`.

    Same optional-version policy as `VectorIndexRequest`/`ChunkRequest`.
    """

    paper_version_id: str | None = None


class ExtractGraphResponse(BaseModel):
    """Response body for `POST /api/v1/papers/{paper_id}/extract-graph`.

    `status` reflects the underlying ingestion job status (`graph_indexing`
    once extraction succeeds) -- this never implies Neo4j has been written
    to; see docs/ARCHITECTURE.md's Scientific Knowledge Extraction Layer.
    """

    ingestion_job_id: str
    paper_id: str
    paper_version_id: str
    status: str
    entity_count: int
    relationship_count: int
    unresolved_citation_count: int
    extraction_version: str
    extraction_config_fingerprint: str
    graph_extraction_generation_fingerprint: str
    artifact_reused: bool


@router.post("/{paper_id}/extract-graph", response_model=ExtractGraphResponse)
def extract_graph(
    paper_id: str,
    request: ExtractGraphRequest = Body(default_factory=ExtractGraphRequest),
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_db_session),
    llm_provider: LLMProvider = Depends(get_llm_provider),
) -> ExtractGraphResponse:
    """Explicitly extract scientific entities/relationships from a chunked
    paper version's `chunks.json`.

    Requires the paper version to already be `CHUNKED` (prompt #3/#33 --
    extraction always starts from `chunks.json`, never re-parses or
    re-chunks; vector indexing is not a precondition). Never writes to
    Neo4j, and never automatically continues into it.
    """

    service = ScientificKnowledgeExtractionService(
        llm_provider,
        ChunkArtifactStorage(settings),
        GraphExtractionArtifactStorage(settings),
        PaperRepository(session),
        IngestionRepository(session),
        extraction_version=settings.EXTRACTION_VERSION,
    )
    result = service.extract(paper_id, request.paper_version_id)

    return ExtractGraphResponse(
        ingestion_job_id=result.job.ingestion_job_id,
        paper_id=result.job.paper_id,
        paper_version_id=result.job.paper_version_id,
        status=result.job.status.value,
        entity_count=result.entity_count,
        relationship_count=result.relationship_count,
        unresolved_citation_count=result.unresolved_citation_count,
        extraction_version=result.extraction_version,
        extraction_config_fingerprint=result.extraction_config_fingerprint,
        graph_extraction_generation_fingerprint=result.graph_extraction_generation_fingerprint,
        artifact_reused=result.extraction_reused,
    )


class GraphEntityResponse(BaseModel):
    entity_id: str
    entity_type: str
    canonical_name: str
    aliases: list[str]
    metadata: dict[str, Any]


class GraphRelationshipResponse(BaseModel):
    relationship_id: str
    source_entity_id: str
    relationship_type: str
    target_entity_id: str
    confidence: float
    source_chunk_id: str | None
    extraction_version: str | None
    metadata: dict[str, Any]


class UnresolvedCitationResponse(BaseModel):
    raw_text: str
    source_chunk_id: str
    reason: str


class GraphExtractionWarningResponse(BaseModel):
    code: str
    detail: str
    source_chunk_id: str | None


class GraphExtractionInspectionResponse(BaseModel):
    """Response body for `GET /api/v1/papers/{paper_id}/graph-extraction`.

    This is the extraction artifact, not a Neo4j graph (prompt #64) --
    Neo4j is not written to at this stage.
    """

    paper_id: str
    paper_version_id: str
    extraction_version: str
    generation_fingerprint: str
    entity_count: int
    relationship_count: int
    entities: list[GraphEntityResponse]
    relationships: list[GraphRelationshipResponse]
    unresolved_citations: list[UnresolvedCitationResponse]
    warnings: list[GraphExtractionWarningResponse]


@router.get("/{paper_id}/graph-extraction", response_model=GraphExtractionInspectionResponse)
def get_graph_extraction(
    paper_id: str,
    paper_version_id: str | None = Query(None),
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_db_session),
) -> GraphExtractionInspectionResponse:
    """Inspect the graph extraction artifact for a paper version
    (development/demo). Entities/relationships cannot be created or edited
    via this API -- they are deterministic derived artifacts; rerun
    `/extract-graph` with a new configuration/version to regenerate them.
    """

    paper_repo = PaperRepository(session)
    paper = resolve_paper(paper_repo, paper_id)
    version = resolve_version(paper_repo, paper, paper_version_id)

    if not version.graph_extraction_artifact_path:
        raise GraphExtractionArtifactNotFoundError(
            f"paper version {version.paper_version_id} has not been graph-extracted yet"
        )

    artifact = GraphExtractionArtifactStorage(settings).try_read(
        source=paper.source, source_id=paper.source_id, version=version.version
    )
    if artifact is None:
        raise GraphExtractionArtifactNotFoundError(
            f"graph extraction artifact for {version.paper_version_id} is missing or corrupt"
        )

    return GraphExtractionInspectionResponse(
        paper_id=artifact.paper_id,
        paper_version_id=artifact.paper_version_id,
        extraction_version=artifact.extraction.version,
        generation_fingerprint=artifact.extraction.generation_fingerprint,
        entity_count=len(artifact.entities),
        relationship_count=len(artifact.relationships),
        entities=[
            GraphEntityResponse(
                entity_id=entity.entity_id,
                entity_type=entity.entity_type.value,
                canonical_name=entity.canonical_name,
                aliases=entity.aliases,
                metadata=entity.metadata,
            )
            for entity in artifact.entities
        ],
        relationships=[
            GraphRelationshipResponse(
                relationship_id=relationship.relationship_id,
                source_entity_id=relationship.source_entity_id,
                relationship_type=relationship.relationship_type.value,
                target_entity_id=relationship.target_entity_id,
                confidence=relationship.confidence,
                source_chunk_id=relationship.source_chunk_id,
                extraction_version=relationship.extraction_version,
                metadata=relationship.metadata,
            )
            for relationship in artifact.relationships
        ],
        unresolved_citations=[
            UnresolvedCitationResponse(
                raw_text=citation.raw_text,
                source_chunk_id=citation.source_chunk_id,
                reason=citation.reason,
            )
            for citation in artifact.unresolved_citations
        ],
        warnings=[
            GraphExtractionWarningResponse(
                code=warning.code.value, detail=warning.detail, source_chunk_id=warning.source_chunk_id
            )
            for warning in artifact.warnings
        ],
    )


class GraphIndexRequest(BaseModel):
    """Request body for `POST /api/v1/papers/{paper_id}/graph-index`.

    Same optional-version policy as `ExtractGraphRequest`/`VectorIndexRequest`.
    """

    paper_version_id: str | None = None


class GraphIndexResponse(BaseModel):
    """Response body for `POST /api/v1/papers/{paper_id}/graph-index`."""

    ingestion_job_id: str
    paper_id: str
    paper_version_id: str
    status: str
    canonical_entity_count: int
    graph_relationship_count: int
    canonicalization_config_fingerprint: str
    graph_index_generation_fingerprint: str
    unresolved_citations_skipped: int
    index_reused: bool


@router.post("/{paper_id}/graph-index", response_model=GraphIndexResponse)
def graph_index_paper(
    paper_id: str,
    request: GraphIndexRequest = Body(default_factory=GraphIndexRequest),
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_db_session),
    graph_repository: GraphRepository = Depends(get_graph_repository),
) -> GraphIndexResponse:
    """Explicitly canonicalize a graph-extracted paper version's entities/
    relationships and index them into Neo4j.

    Requires the paper version to already have a valid `graph_extraction.json`
    (prompt #37 -- indexing always starts from the extraction artifact,
    never re-extracts). Never calls an LLM, never triggers retrieval, and
    never automatically continues into `READY`.
    """

    service = GraphIndexingService(
        graph_repository,
        GraphExtractionArtifactStorage(settings),
        PaperRepository(session),
        IngestionRepository(session),
        canonicalization_version=settings.CANONICALIZATION_VERSION,
        neo4j_database=settings.NEO4J_DATABASE,
    )
    result = service.index_paper_version(paper_id, request.paper_version_id)

    return GraphIndexResponse(
        ingestion_job_id=result.job.ingestion_job_id,
        paper_id=result.job.paper_id,
        paper_version_id=result.job.paper_version_id,
        status=result.job.status.value,
        canonical_entity_count=result.canonical_entity_count,
        graph_relationship_count=result.graph_relationship_count,
        canonicalization_config_fingerprint=result.canonicalization_config_fingerprint,
        graph_index_generation_fingerprint=result.graph_index_generation_fingerprint,
        unresolved_citations_skipped=result.unresolved_citations_skipped,
        index_reused=result.index_reused,
    )
