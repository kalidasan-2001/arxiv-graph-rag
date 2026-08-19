"""PostgreSQL ORM models for the metadata and ingestion control plane.

These map to `papers`, `paper_versions`, `ingestion_jobs`, and
`ingestion_steps`. They intentionally do NOT inherit from or mix with
`app.domain` models (CLAUDE.md-aligned rule for this stage) -- conversion
happens only through `app/storage/postgres/mappings.py`.

No `paper_sections` / `paper_chunks` tables yet: those belong to the
parsing/chunking stage, out of scope here.
"""

from datetime import datetime

from sqlalchemy import BigInteger, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.storage.postgres.base import Base


class PaperRecord(Base):
    """A logical scientific paper. `(source, source_id)` is the natural key
    that `paper_id` (a stable domain identifier) is derived from."""

    __tablename__ = "papers"
    __table_args__ = (
        UniqueConstraint("source", "source_id", name="uq_papers_source_source_id"),
    )

    paper_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    source_id: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    authors: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    categories: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    published_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    updated_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    pdf_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    record_updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    versions: Mapped[list["PaperVersionRecord"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )


class PaperVersionRecord(Base):
    """One immutable revision of a `PaperRecord` (e.g. arXiv v1, v2, v3)."""

    __tablename__ = "paper_versions"
    __table_args__ = (
        UniqueConstraint(
            "paper_id", "version", name="uq_paper_versions_paper_id_version"
        ),
    )

    paper_version_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    paper_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("papers.paper_id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    source_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Always a SHA-256 hex digest of the stored raw PDF (Prompt 4) -- see
    # `app.domain.papers.PaperVersion` docstring for why there's no
    # separate checksum_algorithm column.
    checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Populated by Prompt 5 (was an unused Prompt-1 placeholder before that).
    parser_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    storage_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    downloaded_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    parsed_artifact_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    parsed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    parser_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    warning_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunked_artifact_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    chunked_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunking_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # SHA-256 of chunks.json itself -- distinct from `checksum` (the raw
    # PDF's) and never overloaded with it (prompt #23).
    chunk_artifact_checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Fingerprint of the *complete effective* chunking configuration
    # (prompt 6.1) -- what reuse/invalidation actually compares, since
    # `chunking_version` alone is a human-maintained label nothing forces
    # a developer to bump when e.g. chunk size or overlap changes.
    chunk_config_fingerprint: Mapped[str | None] = mapped_column(String(128), nullable=True)
    vector_indexed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    vector_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding_provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    embedding_config_fingerprint: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # SHA-256(chunk_artifact_checksum + embedding_config_fingerprint) --
    # the actual reuse/invalidation identity for a paper version's current
    # set of vectors (prompt #25/#28). Also stamped into every Qdrant
    # point's payload, so Qdrant state can be verified independently of
    # what this column claims (prompt #29/#31).
    vector_generation_fingerprint: Mapped[str | None] = mapped_column(String(128), nullable=True)
    qdrant_collection: Mapped[str | None] = mapped_column(String(255), nullable=True)
    graph_extraction_artifact_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    graph_extracted_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    entity_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    relationship_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extraction_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    extraction_config_fingerprint: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # SHA-256(chunk_artifact_checksum + extraction_config_fingerprint) --
    # the actual reuse/invalidation identity for a paper version's current
    # extraction (prompt #23/#40/#41), mirroring `vector_generation_fingerprint`.
    graph_extraction_generation_fingerprint: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )
    # SHA-256 of graph_extraction.json itself -- distinct from
    # `chunk_artifact_checksum` and never overloaded with it (prompt #29).
    graph_extraction_artifact_checksum: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )
    graph_indexed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    canonical_entity_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    graph_relationship_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Fingerprint of the complete effective canonicalization configuration
    # (Prompt 9: version + normalization algorithm + alias registry
    # content + ontology version) -- mirrors `chunk_config_fingerprint`'s
    # role for chunking.
    canonicalization_config_fingerprint: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )
    # SHA-256(graph_extraction_artifact_checksum + canonicalization_config_fingerprint)
    # -- the actual reuse/invalidation identity for a paper version's
    # current Neo4j graph generation (prompt #31), mirroring
    # `graph_extraction_generation_fingerprint`. Also stamped onto every
    # Neo4j relationship this generation writes, so Neo4j state can be
    # verified independently of what this column claims.
    graph_index_generation_fingerprint: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )
    neo4j_database: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    record_updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    paper: Mapped["PaperRecord"] = relationship(back_populates="versions")
    ingestion_jobs: Mapped[list["IngestionJobRecord"]] = relationship(
        back_populates="paper_version", cascade="all, delete-orphan"
    )


class IngestionJobRecord(Base):
    """One operational run of the ingestion pipeline for a paper version."""

    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        Index(
            "ix_ingestion_jobs_paper_version_status", "paper_version_id", "status"
        ),
        # Enforces "at most one active job per paper version" at the
        # database level (CLAUDE.md #28: application logic alone is not
        # enough to protect against a concurrent duplicate). The status
        # values are literal here, not `IngestionStatus` members, because
        # a partial index condition can't reference Python code -- they
        # must be kept in sync with `app.domain.ingestion.TERMINAL_STATUSES`.
        Index(
            "uq_ingestion_jobs_one_active_per_version",
            "paper_version_id",
            unique=True,
            postgresql_where=text("status NOT IN ('ready', 'failed')"),
        ),
    )

    ingestion_job_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    paper_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("papers.paper_id", ondelete="CASCADE"), nullable=False
    )
    paper_version_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("paper_versions.paper_version_id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    failed_stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Records which build of the (not-yet-implemented) ingestion pipeline
    # produced this job, so future reprocessing decisions (CLAUDE.md #55)
    # can tell which jobs ran under an outdated pipeline version.
    pipeline_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    paper_version: Mapped["PaperVersionRecord"] = relationship(
        back_populates="ingestion_jobs"
    )
    steps: Mapped[list["IngestionStepRecord"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class IngestionStepRecord(Base):
    """A single attempt of one processing stage within an ingestion job.

    `step_id` is a plain auto-incrementing integer, not a domain identifier
    from `app.domain.ids` -- steps are Postgres-local operational records,
    never referenced from Qdrant or Neo4j, so they don't need a
    cross-store-stable identity.
    """

    __tablename__ = "ingestion_steps"
    __table_args__ = (
        UniqueConstraint(
            "ingestion_job_id",
            "stage",
            "attempt",
            name="uq_ingestion_steps_job_stage_attempt",
        ),
        Index("ix_ingestion_steps_job_stage", "ingestion_job_id", "stage"),
    )

    step_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ingestion_job_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("ingestion_jobs.ingestion_job_id", ondelete="CASCADE"),
        nullable=False,
    )
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    started_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    step_metadata: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )

    job: Mapped["IngestionJobRecord"] = relationship(back_populates="steps")
