"""Application exception hierarchy.

A small, flat hierarchy that later stages (ingestion, retrieval, reasoning)
can extend. Kept intentionally minimal for this foundation step.
"""


class ApplicationError(Exception):
    """Base class for all application-raised errors."""


class ConfigurationError(ApplicationError):
    """Raised when application configuration is invalid or missing."""


class ExternalServiceError(ApplicationError):
    """Raised when a call to an external/downstream service fails.

    Reserved for future integrations (Qdrant, Neo4j, LLM and embedding
    providers). PostgreSQL, arXiv metadata search, and PDF acquisition are
    now integrated, but their failures are represented by the more
    specific errors below, not this one directly.
    """


class ArxivServiceError(ExternalServiceError):
    """Raised when arXiv is unavailable or returns an unexpected response."""


class ArxivTimeoutError(ArxivServiceError):
    """Raised when a request to arXiv exceeds `ARXIV_REQUEST_TIMEOUT_SECONDS`."""


class ArxivRateLimitError(ArxivServiceError):
    """Raised when arXiv responds with HTTP 429 after exhausting retries."""


class ArxivResponseError(ArxivServiceError):
    """Raised when arXiv's response cannot be parsed as a valid Atom feed."""


class InvalidSearchQueryError(ApplicationError):
    """Raised when a paper search request is invalid.

    E.g. a blank query, `max_results` outside configured bounds, an
    inverted date range, or an unsupported sort option. Not an
    `ExternalServiceError` -- this is an input-validation failure, not an
    arXiv failure.
    """


class PaperNotFoundError(ApplicationError):
    """Raised when a `paper_id` does not exist in the persistence layer."""


class IngestionJobNotFoundError(ApplicationError):
    """Raised when an `ingestion_job_id` does not exist in the persistence layer."""


class InvalidIngestionTransitionError(ApplicationError):
    """Raised when an ingestion status transition is not permitted.

    Raised by the pure domain state machine (`app/domain/ingestion.py`),
    not by the storage layer, so transition rules stay testable without a
    database.
    """


class PersistenceConflictError(ApplicationError):
    """Raised when a write would conflict with an existing, immutable record.

    For example: re-recording a paper version with a different checksum
    than the one already stored for that version.
    """


class PaperVersionNotFoundError(ApplicationError):
    """Raised when a requested/resolved `paper_version_id` does not exist."""


class PdfDownloadError(ExternalServiceError):
    """Raised when downloading a PDF artifact fails (network/HTTP)."""


class PdfTimeoutError(PdfDownloadError):
    """Raised when a PDF download exceeds `PDF_DOWNLOAD_TIMEOUT_SECONDS`."""


class PdfTooLargeError(PdfDownloadError):
    """Raised when a PDF download exceeds `MAX_PAPER_SIZE_MB`. Never retried."""


class PdfNotFoundError(PdfDownloadError):
    """Raised when the source returns HTTP 404 for the PDF URL. Never retried."""


class InvalidPdfError(ApplicationError):
    """Raised when a downloaded file does not look like a valid PDF.

    Not an `ExternalServiceError` -- the request succeeded, but the body
    wasn't a PDF (e.g. arXiv returned an HTML error page with HTTP 200).
    """


class UnsafePdfUrlError(ApplicationError):
    """Raised when a PDF URL is missing, uses an unsupported scheme, or its
    host is not in the trusted allowlist (CLAUDE.md #36; this stage's SSRF
    prevention requirement)."""


class InvalidStoragePathError(ApplicationError):
    """Raised when a storage path component fails path-safety validation.

    Should be unreachable in normal operation -- domain identifiers are
    already validated before they ever reach `PaperStorage` -- this is
    defense in depth, not an expected user-facing condition.
    """


class PdfParseError(ApplicationError):
    """Raised when the parser fails to extract usable text/structure from a PDF."""


class UnsupportedPdfError(PdfParseError):
    """Raised when the PDF cannot be opened at all: corrupt, encrypted, or not a real PDF."""


class ParseArtifactNotFoundError(ApplicationError):
    """Raised when a parsed document is requested but no successful parse exists.

    Used by the read-only inspection endpoint (`GET .../document`), and by
    `ChunkingService` as the precondition check for "there is no valid
    parsed source to chunk from" -- during parsing itself, a missing/
    corrupt `parsed.json` is a reconciliation trigger (reparse), not an
    error.
    """


class ChunkingError(ApplicationError):
    """Raised when the chunker produces invalid output (prompt #35): a
    chunker bug, not user error -- fails loudly rather than persisting
    inconsistent chunk data."""


class ChunkArtifactNotFoundError(ApplicationError):
    """Raised when chunks are requested but no successful chunking run exists.

    Used by the read-only inspection endpoint (`GET .../chunks`), not by
    the chunking service itself -- there, a missing/corrupt `chunks.json`
    is a reconciliation trigger (rechunk), not an error. Also used by
    `VectorIndexingService` as the precondition check for "there is no
    valid chunk source to index from" (prompt 7 mirrors prompt 6's
    `ParseArtifactNotFoundError` reuse for chunking).
    """


class EmbeddingProviderError(ApplicationError):
    """Raised when an embedding provider fails or returns malformed output.

    Base class for embedding-provider failures; also raised directly for
    output that fails validation in a way not specific to dimension (e.g.
    a NaN/infinite value) -- a provider bug or an unexpected upstream
    response, not user error.
    """


class EmbeddingModelLoadError(EmbeddingProviderError):
    """Raised when the configured embedding model fails to load.

    E.g. no network access on first use (models download from Hugging
    Face on demand) or an invalid `EMBEDDING_MODEL` identifier.
    """


class EmbeddingDimensionError(EmbeddingProviderError):
    """Raised when embedding output shape is wrong: the number of vectors
    doesn't match the number of input texts, or a vector's length doesn't
    match `EmbeddingProvider.dimension`."""


class VectorIndexingError(ApplicationError):
    """Raised when the chunk artifact isn't valid to index from, or a
    vector-indexing invariant fails (e.g. post-upsert verification finds
    fewer points than expected) -- an indexing bug or a stale/legacy
    artifact, not user error."""


class VectorCollectionIncompatibleError(ApplicationError):
    """Raised when an existing Qdrant collection's dimension/distance
    metric doesn't match the active embedding provider.

    Never silently deleted/recreated (prompt #14) -- a dimension or
    metric mismatch means the collection was built for a different
    embedding configuration, and destroying it would lose every other
    paper's vectors too.
    """


class VectorStoreUnavailableError(ExternalServiceError):
    """Raised when Qdrant is unreachable or returns an unexpected response."""


class VectorSearchError(ApplicationError):
    """Raised when a semantic search request is invalid (e.g. `top_k`
    outside configured bounds) or the underlying Qdrant query fails."""


class HybridRetrievalError(ApplicationError):
    """Raised when an explicit retrieval request is invalid or a requested branch fails."""


class QueryPlanningError(ApplicationError):
    """Raised when query analysis or retrieval planning cannot produce a valid plan."""


class GraphSearchError(ApplicationError):
    """Raised when a deterministic graph search request is invalid."""


class LLMProviderError(ExternalServiceError):
    """Raised when an LLM provider call fails or returns something unusable."""


class LLMTimeoutError(LLMProviderError):
    """Raised when an LLM request exceeds `LLM_TIMEOUT_SECONDS` after
    exhausting `LLM_MAX_RETRIES`."""


class LLMResponseError(LLMProviderError):
    """Raised when the LLM's response cannot be parsed as JSON, or fails
    structured-schema validation, after exhausting `LLM_MAX_RETRIES`
    (prompt #38/#59) -- malformed output is retried a bounded number of
    times, not trusted on the first attempt and not retried forever."""


class GraphExtractionError(ApplicationError):
    """Raised when the chunk artifact isn't valid to extract from, or a
    graph-extraction invariant fails -- mirrors `VectorIndexingError`."""


class GraphExtractionArtifactNotFoundError(ApplicationError):
    """Raised when a graph extraction result is requested but no
    successful extraction run exists.

    Used by the read-only inspection endpoint, not by the extraction
    service itself -- there, a missing/corrupt `graph_extraction.json` is
    a reconciliation trigger (re-extract), not an error. Also used by
    `GraphIndexingService` as the precondition check for "there is no
    valid extraction artifact to canonicalize/index from" (Prompt 9
    mirrors Prompt 7's `ChunkArtifactNotFoundError` reuse for indexing).
    """


class GraphIndexingError(ApplicationError):
    """Raised when the extraction artifact isn't valid to canonicalize/index
    from, or a graph-indexing invariant fails (e.g. post-upsert
    verification finds a different relationship-id set than expected) --
    mirrors `VectorIndexingError`/`GraphExtractionError`."""


class GraphStoreUnavailableError(ExternalServiceError):
    """Raised when Neo4j is unreachable or returns an unexpected response.

    Never silently caught to "continue without the graph" (CLAUDE.md #43,
    no silent fallbacks) -- a paper's ingestion job simply cannot reach
    `GRAPH_INDEXED` while this is being raised.
    """


class GraphNotFoundError(ApplicationError):
    """Raised when a graph inspection endpoint (`GET .../graph/papers/{id}`,
    `GET .../graph/entities/{id}`) is asked about a paper/entity that has
    no data in Neo4j yet. Read-only inspection only -- never raised by
    `GraphIndexingService` itself."""


class GraphEntityNotFoundError(GraphSearchError):
    """Raised when graph retrieval cannot resolve the requested start entity."""


class GraphEntityAmbiguousError(GraphSearchError):
    """Raised when name-based graph entity lookup has multiple candidates."""

    def __init__(self, message: str, candidates: list[dict] | None = None) -> None:
        super().__init__(message)
        self.candidates = candidates or []


class EvidenceProvenanceError(ApplicationError):
    """Base class for deterministic evidence provenance failures."""


class EvidenceGenerationMismatchError(EvidenceProvenanceError):
    """Raised when exact source evidence resolves to an unexpected generation."""


class EvidenceIdentityMismatchError(EvidenceProvenanceError):
    """Raised when exact source evidence resolves to the wrong stable identity."""
