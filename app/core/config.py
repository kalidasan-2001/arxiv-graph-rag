"""Application configuration.

All configuration is loaded from environment variables (optionally via a
``.env`` file) using ``pydantic-settings``. This module intentionally only
defines *placeholders* for future infrastructure (database, vector store,
graph store, LLM/embedding providers). No connections are established here
or anywhere else at this stage.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings.

    Values are resolved from environment variables first, falling back to
    the defaults declared below, with an optional local ``.env`` file used
    for development convenience.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    APP_NAME: str = "arxiv-graph-rag-api"
    APP_ENV: str = "development"
    APP_DEBUG: bool = False

    # --- API server ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    FRONTEND_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- PostgreSQL: metadata + ingestion control plane ---
    DATABASE_URL: str | None = None

    # --- arXiv discovery ---
    ARXIV_REQUEST_TIMEOUT_SECONDS: float = 15
    ARXIV_DEFAULT_MAX_RESULTS: int = 20
    ARXIV_MAX_RESULTS_LIMIT: int = 100
    ARXIV_MAX_RETRIES: int = 2

    # --- Raw PDF acquisition + storage ---
    PAPER_STORAGE_PATH: str = "./data/papers"
    MAX_PAPER_SIZE_MB: int = 50
    PDF_DOWNLOAD_TIMEOUT_SECONDS: float = 30
    PDF_DOWNLOAD_MAX_RETRIES: int = 2

    # --- Section-aware chunking ---
    # Bumping CHUNKING_VERSION signals that vector re-indexing may be
    # required (prompt #7) -- it's baked into every chunk_id, not just
    # metadata, so a config change can never silently collide with the
    # previous run's chunk identities.
    CHUNKING_VERSION: str = "v1"
    CHUNK_SIZE_TOKENS: int = 700
    CHUNK_OVERLAP_TOKENS: int = 100
    MIN_CHUNK_TOKENS: int = 80

    # --- Qdrant vector store (Prompt 7) ---
    QDRANT_URL: str | None = "http://localhost:6333"
    QDRANT_COLLECTION: str = "scientific_chunks"
    # Optional: local Docker Qdrant has no auth by default (prompt #10).
    QDRANT_API_KEY: str | None = None
    QDRANT_TIMEOUT_SECONDS: float = 30

    # --- Neo4j knowledge graph (Prompt 9) ---
    # URI/USERNAME/PASSWORD existed as unconnected Prompt-0 placeholders;
    # this stage is what actually connects them.
    NEO4J_URI: str | None = None
    NEO4J_USERNAME: str | None = None
    NEO4J_PASSWORD: str | None = None
    NEO4J_DATABASE: str = "neo4j"
    NEO4J_TIMEOUT_SECONDS: float = 30

    # --- Canonical entity resolution (Prompt 9) ---
    # Bumping CANONICALIZATION_VERSION signals that re-indexing may be
    # required -- feeds `canonicalization_config_fingerprint`, mirroring
    # EXTRACTION_VERSION's role in extraction identity (prompt #32/#33).
    CANONICALIZATION_VERSION: str = "v1"

    # --- LLM provider (Prompt 8) ---
    # V1's only implementation talks the OpenAI-compatible chat-completions
    # HTTP API -- this is what lets LM Studio, OpenAI itself, and other
    # OpenAI-compatible endpoints all work through one provider (prompt #43),
    # without hard-wiring this project to a single cloud vendor's SDK.
    LLM_PROVIDER: str = "openai_compatible"
    LLM_MODEL: str | None = None
    # No default: unlike Qdrant/embeddings (safe, free local defaults),
    # silently defaulting this to a real cloud endpoint could send paper
    # text to a paid third-party service the user never opted into
    # (CLAUDE.md #29 -- fail clearly when required configuration is missing).
    LLM_BASE_URL: str | None = None
    LLM_API_KEY: str | None = None
    LLM_TIMEOUT_SECONDS: float = 60
    LLM_MAX_RETRIES: int = 2
    # Structured extraction, not creative generation (prompt #24) -- as
    # deterministic as the provider allows.
    LLM_TEMPERATURE: float = 0.0

    # --- Scientific knowledge extraction (Prompt 8) ---
    # Bumping EXTRACTION_VERSION signals that re-extraction may be required
    # -- feeds `extraction_config_fingerprint`, mirroring CHUNKING_VERSION's
    # role in chunk identity (prompt #21/#22).
    EXTRACTION_VERSION: str = "v1"

    # --- Embedding provider (Prompt 7) ---
    # V1's only implementation; the abstraction (`EmbeddingProvider`) is
    # what the rest of the app depends on, not this string (CLAUDE.md #16).
    EMBEDDING_PROVIDER: str = "sentence_transformers"
    # all-MiniLM-L6-v2: 384-dim, ~80MB, CPU-friendly, a well-established
    # default for semantic search -- deliberately configurable, not
    # hard-coded into the provider (prompt #4/#5).
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_NORMALIZE: bool = True

    # --- Vector search (Prompt 7) ---
    VECTOR_SEARCH_DEFAULT_TOP_K: int = 5
    VECTOR_SEARCH_MAX_TOP_K: int = 50

    # --- Graph retrieval (Prompt 10) ---
    GRAPH_MAX_DEPTH: int = 3
    GRAPH_DEFAULT_LIMIT: int = 20
    GRAPH_MAX_LIMIT: int = 100
    EVIDENCE_MAX_SUPPORTING_CHUNKS: int = 5

    # --- Hybrid retrieval (Prompt 12) ---
    HYBRID_RRF_K: int = 60
    HYBRID_DEFAULT_TOP_K: int = 10
    HYBRID_MAX_TOP_K: int = 50

    # --- Query analysis / retrieval planning (Prompt 14) ---
    QUERY_ANALYSIS_PROMPT_VERSION: str = "v1"
    QUERY_ANALYSIS_SCHEMA_VERSION: str = "v1"
    QUERY_PLANNER_VERSION: str = "v1"
    QUERY_PLANNER_RULES_VERSION: str = "v1"

    # --- Evidence sufficiency / bounded refinement (Prompt 16) ---
    MAX_RETRIEVAL_ROUNDS: int = 2
    EVIDENCE_CRITIC_PROMPT_VERSION: str = "v1"
    EVIDENCE_CRITIC_SCHEMA_VERSION: str = "v1"
    EVIDENCE_CRITIC_RULES_VERSION: str = "v1"

    # --- Grounded answer generation (Prompt 17) ---
    ANSWER_GENERATION_PROMPT_VERSION: str = "v1"
    ANSWER_GENERATION_SCHEMA_VERSION: str = "v1"
    ANSWER_CONTEXT_BUILDER_VERSION: str = "v1"
    ANSWER_MAX_EVIDENCE_ITEMS: int = 10
    ANSWER_MAX_CONTEXT_CHARS: int = 30000
    ANSWER_MAX_OUTPUT_TOKENS: int = 800

    # --- Citation validation (Prompt 18) ---
    CITATION_VALIDATOR_VERSION: str = "v1"
    CITATION_MARKER_SCHEMA_VERSION: str = "v1"

    # --- Final grounding / confidence gates (Prompt 19) ---
    GROUNDING_RULES_VERSION: str = "v1"
    CONFIDENCE_RULES_VERSION: str = "v1"
    ABSTENTION_TEMPLATE_VERSION: str = "v1"


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance.

    Cached so environment variables are parsed once per process rather than
    on every access.
    """

    return Settings()
