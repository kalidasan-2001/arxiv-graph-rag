"""Application entry point and factory.

Run locally with:

    uvicorn app.main:app --reload
"""

import logging

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.graph import router as graph_router
from app.api.routes.health import router as health_router
from app.api.routes.ingestion import router as ingestion_router
from app.api.routes.papers import router as papers_router
from app.api.routes.query import router as query_router
from app.api.routes.search import router as search_router
from app.core.config import get_settings
from app.core.exceptions import (
    ApplicationError,
    ArxivServiceError,
    ArxivTimeoutError,
    ChunkArtifactNotFoundError,
    ChunkingError,
    EmbeddingModelLoadError,
    EmbeddingProviderError,
    EvidenceProvenanceError,
    GraphExtractionArtifactNotFoundError,
    GraphExtractionError,
    GraphEntityAmbiguousError,
    GraphEntityNotFoundError,
    GraphIndexingError,
    GraphNotFoundError,
    GraphSearchError,
    GraphStoreUnavailableError,
    HybridRetrievalError,
    IngestionJobNotFoundError,
    InvalidIngestionTransitionError,
    InvalidPdfError,
    InvalidSearchQueryError,
    LLMProviderError,
    LLMResponseError,
    LLMTimeoutError,
    PaperNotFoundError,
    PaperVersionNotFoundError,
    ParseArtifactNotFoundError,
    PdfDownloadError,
    PdfNotFoundError,
    PdfParseError,
    PdfTimeoutError,
    UnsafePdfUrlError,
    UnsupportedPdfError,
    VectorCollectionIncompatibleError,
    VectorIndexingError,
    VectorSearchError,
    VectorStoreUnavailableError,
)
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""

    settings = get_settings()
    configure_logging(settings)

    app = FastAPI(
        title="ArXiv Hybrid Graph-RAG API",
        debug=settings.APP_DEBUG,
    )
    frontend_origins = [origin.strip() for origin in settings.FRONTEND_ORIGINS.split(",") if origin.strip()]
    if frontend_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=frontend_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    app.include_router(health_router, prefix="/api/v1")
    app.include_router(papers_router, prefix="/api/v1")
    app.include_router(ingestion_router, prefix="/api/v1")
    app.include_router(search_router, prefix="/api/v1")
    app.include_router(query_router, prefix="/api/v1")
    app.include_router(graph_router, prefix="/api/v1")

    @app.exception_handler(InvalidSearchQueryError)
    async def invalid_search_query_handler(
        request: Request, exc: InvalidSearchQueryError
    ) -> JSONResponse:
        """Bad input (blank query, out-of-range max_results, ...) -> 400."""

        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ArxivTimeoutError)
    async def arxiv_timeout_handler(request: Request, exc: ArxivTimeoutError) -> JSONResponse:
        """arXiv did not respond in time -> 504."""

        logger.warning("arXiv request timed out: %s", exc)
        return JSONResponse(status_code=504, content={"detail": "arXiv request timed out."})

    @app.exception_handler(ArxivServiceError)
    async def arxiv_service_error_handler(
        request: Request, exc: ArxivServiceError
    ) -> JSONResponse:
        """arXiv unreachable, rate-limited, or returned a bad response -> 503."""

        logger.warning("arXiv service error: %s", exc)
        return JSONResponse(
            status_code=503, content={"detail": "arXiv is temporarily unavailable."}
        )

    @app.exception_handler(PaperNotFoundError)
    async def paper_not_found_handler(request: Request, exc: PaperNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(PaperVersionNotFoundError)
    async def paper_version_not_found_handler(
        request: Request, exc: PaperVersionNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(IngestionJobNotFoundError)
    async def ingestion_job_not_found_handler(
        request: Request, exc: IngestionJobNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(InvalidIngestionTransitionError)
    async def invalid_ingestion_transition_handler(
        request: Request, exc: InvalidIngestionTransitionError
    ) -> JSONResponse:
        """A state-machine conflict (e.g. a stale concurrent request) -> 409."""

        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(UnsafePdfUrlError)
    async def unsafe_pdf_url_handler(request: Request, exc: UnsafePdfUrlError) -> JSONResponse:
        """Missing/untrusted PDF URL -- SSRF prevention (prompt #17) -> 400."""

        logger.warning("rejected unsafe PDF URL: %s", exc)
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(PdfTimeoutError)
    async def pdf_timeout_handler(request: Request, exc: PdfTimeoutError) -> JSONResponse:
        logger.warning("PDF download timed out: %s", exc)
        return JSONResponse(status_code=504, content={"detail": "PDF download timed out."})

    @app.exception_handler(PdfNotFoundError)
    async def pdf_not_found_handler(request: Request, exc: PdfNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404, content={"detail": "PDF not found at the paper's known URL."}
        )

    @app.exception_handler(PdfDownloadError)
    async def pdf_download_error_handler(request: Request, exc: PdfDownloadError) -> JSONResponse:
        """Network failure, oversized file, or transient upstream error -> 502."""

        logger.warning("PDF download failed: %s", exc)
        return JSONResponse(
            status_code=502, content={"detail": "Failed to download the PDF from the source."}
        )

    @app.exception_handler(InvalidPdfError)
    async def invalid_pdf_handler(request: Request, exc: InvalidPdfError) -> JSONResponse:
        """The source responded, but the body wasn't a valid PDF -> 502."""

        logger.warning("downloaded file failed PDF validation: %s", exc)
        return JSONResponse(
            status_code=502, content={"detail": "The downloaded file was not a valid PDF."}
        )

    @app.exception_handler(ParseArtifactNotFoundError)
    async def parse_artifact_not_found_handler(
        request: Request, exc: ParseArtifactNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(UnsupportedPdfError)
    async def unsupported_pdf_handler(request: Request, exc: UnsupportedPdfError) -> JSONResponse:
        """The stored PDF can't even be opened (corrupt/encrypted) -> 422."""

        logger.warning("PDF could not be opened for parsing: %s", exc)
        return JSONResponse(
            status_code=422, content={"detail": "The PDF could not be opened for parsing."}
        )

    @app.exception_handler(PdfParseError)
    async def pdf_parse_error_handler(request: Request, exc: PdfParseError) -> JSONResponse:
        """The parser opened the PDF but couldn't extract usable content -> 422."""

        logger.warning("PDF parsing failed: %s", exc)
        return JSONResponse(status_code=422, content={"detail": "Failed to parse the PDF."})

    @app.exception_handler(ChunkArtifactNotFoundError)
    async def chunk_artifact_not_found_handler(
        request: Request, exc: ChunkArtifactNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ChunkingError)
    async def chunking_error_handler(request: Request, exc: ChunkingError) -> JSONResponse:
        """The chunker produced invalid output -- a chunker bug, not user error -> 422."""

        logger.error("chunking validation failed: %s", exc, exc_info=exc)
        return JSONResponse(status_code=422, content={"detail": "Failed to chunk the document."})

    @app.exception_handler(EmbeddingModelLoadError)
    async def embedding_model_load_handler(
        request: Request, exc: EmbeddingModelLoadError
    ) -> JSONResponse:
        """Model failed to load (e.g. no network on first download) -> 503."""

        logger.warning("embedding model load failed: %s", exc)
        return JSONResponse(
            status_code=503, content={"detail": "The embedding model is unavailable."}
        )

    @app.exception_handler(EmbeddingProviderError)
    async def embedding_provider_error_handler(
        request: Request, exc: EmbeddingProviderError
    ) -> JSONResponse:
        """The embedding provider produced malformed output -- a provider
        bug, not user error -> 502."""

        logger.error("embedding provider error: %s", exc, exc_info=exc)
        return JSONResponse(
            status_code=502, content={"detail": "The embedding provider returned an error."}
        )

    @app.exception_handler(VectorIndexingError)
    async def vector_indexing_error_handler(
        request: Request, exc: VectorIndexingError
    ) -> JSONResponse:
        """The chunk artifact isn't valid to index, or an indexing
        invariant failed -> 422."""

        logger.error("vector indexing failed: %s", exc, exc_info=exc)
        return JSONResponse(status_code=422, content={"detail": "Failed to index vectors."})

    @app.exception_handler(VectorCollectionIncompatibleError)
    async def vector_collection_incompatible_handler(
        request: Request, exc: VectorCollectionIncompatibleError
    ) -> JSONResponse:
        """The existing Qdrant collection doesn't match the active
        embedding provider -- never auto-deleted/recreated -> 409."""

        logger.error("Qdrant collection incompatible: %s", exc)
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(VectorStoreUnavailableError)
    async def vector_store_unavailable_handler(
        request: Request, exc: VectorStoreUnavailableError
    ) -> JSONResponse:
        """Qdrant is unreachable or returned an unexpected response -> 503."""

        logger.warning("Qdrant unavailable: %s", exc)
        return JSONResponse(
            status_code=503, content={"detail": "Qdrant is temporarily unavailable."}
        )

    @app.exception_handler(VectorSearchError)
    async def vector_search_error_handler(request: Request, exc: VectorSearchError) -> JSONResponse:
        """Bad search input (blank query, out-of-range top_k) -> 400."""

        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(HybridRetrievalError)
    async def hybrid_retrieval_error_handler(
        request: Request, exc: HybridRetrievalError
    ) -> JSONResponse:
        """Bad explicit retrieval request -> 400."""

        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(GraphExtractionArtifactNotFoundError)
    async def graph_extraction_artifact_not_found_handler(
        request: Request, exc: GraphExtractionArtifactNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(GraphExtractionError)
    async def graph_extraction_error_handler(
        request: Request, exc: GraphExtractionError
    ) -> JSONResponse:
        """The chunk artifact isn't valid to extract from, or an
        extraction invariant failed -> 422."""

        logger.error("graph extraction failed: %s", exc, exc_info=exc)
        return JSONResponse(status_code=422, content={"detail": "Failed to extract graph data."})

    @app.exception_handler(GraphIndexingError)
    async def graph_indexing_error_handler(request: Request, exc: GraphIndexingError) -> JSONResponse:
        """The extraction artifact isn't valid to canonicalize/index from,
        or a graph-indexing invariant failed -> 422."""

        logger.error("graph indexing failed: %s", exc, exc_info=exc)
        return JSONResponse(status_code=422, content={"detail": "Failed to index the knowledge graph."})

    @app.exception_handler(GraphStoreUnavailableError)
    async def graph_store_unavailable_handler(
        request: Request, exc: GraphStoreUnavailableError
    ) -> JSONResponse:
        """Neo4j is unreachable or returned an unexpected response -> 503."""

        logger.warning("Neo4j unavailable: %s", exc)
        return JSONResponse(
            status_code=503, content={"detail": "Neo4j is temporarily unavailable."}
        )

    @app.exception_handler(GraphNotFoundError)
    async def graph_not_found_handler(request: Request, exc: GraphNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(GraphEntityNotFoundError)
    async def graph_entity_not_found_handler(
        request: Request, exc: GraphEntityNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(GraphEntityAmbiguousError)
    async def graph_entity_ambiguous_handler(
        request: Request, exc: GraphEntityAmbiguousError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "candidates": exc.candidates},
        )

    @app.exception_handler(GraphSearchError)
    async def graph_search_error_handler(request: Request, exc: GraphSearchError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(EvidenceProvenanceError)
    async def evidence_provenance_error_handler(
        request: Request, exc: EvidenceProvenanceError
    ) -> JSONResponse:
        logger.warning("Evidence provenance validation failed: %s", exc)
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(LLMTimeoutError)
    async def llm_timeout_handler(request: Request, exc: LLMTimeoutError) -> JSONResponse:
        logger.warning("LLM request timed out: %s", exc)
        return JSONResponse(status_code=504, content={"detail": "LLM request timed out."})

    @app.exception_handler(LLMResponseError)
    async def llm_response_error_handler(request: Request, exc: LLMResponseError) -> JSONResponse:
        """The LLM returned unparseable/invalid structured output after
        retries were exhausted -- an upstream quality issue, not user
        error -> 502."""

        logger.warning("LLM returned invalid structured output: %s", exc)
        return JSONResponse(
            status_code=502, content={"detail": "The LLM provider returned invalid output."}
        )

    @app.exception_handler(LLMProviderError)
    async def llm_provider_error_handler(request: Request, exc: LLMProviderError) -> JSONResponse:
        """LLM endpoint unreachable or returned an unexpected response -> 503."""

        logger.warning("LLM provider unavailable: %s", exc)
        return JSONResponse(
            status_code=503, content={"detail": "The LLM provider is temporarily unavailable."}
        )

    @app.exception_handler(ApplicationError)
    async def application_error_handler(
        request: Request, exc: ApplicationError
    ) -> JSONResponse:
        """Return a generic JSON error body for unhandled application errors."""

        logger.error("Unhandled application error: %s", exc, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal application error."},
        )

    logger.info("%s started in '%s' environment", settings.APP_NAME, settings.APP_ENV)
    return app


app = create_app()
