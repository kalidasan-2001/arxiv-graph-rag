"""Explicit mapping functions between ORM records and domain models.

This is the *only* place an ORM record and a domain model may meet.
Domain models never inherit from SQLAlchemy classes, and no SQLAlchemy
object crosses the storage/application boundary (this stage's explicit
architecture rule, consistent with CLAUDE.md #15's provider-boundary rule).
"""

from app.domain.enums import IngestionStatus, ProcessingStage, StepStatus
from app.domain.ingestion import IngestionJob, IngestionStepState
from app.domain.papers import Paper, PaperVersion
from app.storage.postgres.models import (
    IngestionJobRecord,
    IngestionStepRecord,
    PaperRecord,
    PaperVersionRecord,
)

# --- Paper -------------------------------------------------------------


def paper_record_to_domain(record: PaperRecord) -> Paper:
    return Paper(
        paper_id=record.paper_id,
        source=record.source,
        source_id=record.source_id,
        title=record.title,
        abstract=record.abstract,
        authors=list(record.authors or []),
        categories=list(record.categories or []),
        published_at=record.published_at,
        updated_at=record.updated_at,
        pdf_url=record.pdf_url,
    )


def domain_paper_to_record(paper: Paper) -> PaperRecord:
    return PaperRecord(
        paper_id=paper.paper_id,
        source=paper.source,
        source_id=paper.source_id,
        title=paper.title,
        abstract=paper.abstract,
        authors=list(paper.authors),
        categories=list(paper.categories),
        published_at=paper.published_at,
        updated_at=paper.updated_at,
        pdf_url=paper.pdf_url,
    )


def update_paper_record_from_domain(record: PaperRecord, paper: Paper) -> None:
    """Update mutable metadata fields on an existing record in place.

    Identity fields (`paper_id`, `source`, `source_id`) are intentionally
    never reassigned here -- an upsert must never change a paper's identity.
    """

    record.title = paper.title
    record.abstract = paper.abstract
    record.authors = list(paper.authors)
    record.categories = list(paper.categories)
    record.published_at = paper.published_at
    record.updated_at = paper.updated_at
    record.pdf_url = paper.pdf_url


# --- Paper version -------------------------------------------------------


def paper_version_record_to_domain(record: PaperVersionRecord) -> PaperVersion:
    return PaperVersion(
        paper_version_id=record.paper_version_id,
        paper_id=record.paper_id,
        version=record.version,
        source_version=record.source_version,
        checksum=record.checksum,
        parser_version=record.parser_version,
        storage_path=record.storage_path,
        downloaded_at=record.downloaded_at,
        file_size_bytes=record.file_size_bytes,
        parsed_artifact_path=record.parsed_artifact_path,
        parsed_at=record.parsed_at,
        parser_name=record.parser_name,
        page_count=record.page_count,
        section_count=record.section_count,
        warning_count=record.warning_count,
        chunked_artifact_path=record.chunked_artifact_path,
        chunked_at=record.chunked_at,
        chunk_count=record.chunk_count,
        chunking_version=record.chunking_version,
        chunk_artifact_checksum=record.chunk_artifact_checksum,
        chunk_config_fingerprint=record.chunk_config_fingerprint,
        vector_indexed_at=record.vector_indexed_at,
        vector_count=record.vector_count,
        embedding_provider=record.embedding_provider,
        embedding_model=record.embedding_model,
        embedding_config_fingerprint=record.embedding_config_fingerprint,
        vector_generation_fingerprint=record.vector_generation_fingerprint,
        qdrant_collection=record.qdrant_collection,
        graph_extraction_artifact_path=record.graph_extraction_artifact_path,
        graph_extracted_at=record.graph_extracted_at,
        entity_count=record.entity_count,
        relationship_count=record.relationship_count,
        extraction_version=record.extraction_version,
        extraction_config_fingerprint=record.extraction_config_fingerprint,
        graph_extraction_generation_fingerprint=record.graph_extraction_generation_fingerprint,
        graph_extraction_artifact_checksum=record.graph_extraction_artifact_checksum,
        graph_indexed_at=record.graph_indexed_at,
        canonical_entity_count=record.canonical_entity_count,
        graph_relationship_count=record.graph_relationship_count,
        canonicalization_config_fingerprint=record.canonicalization_config_fingerprint,
        graph_index_generation_fingerprint=record.graph_index_generation_fingerprint,
        neo4j_database=record.neo4j_database,
        created_at=record.created_at,
    )


def domain_paper_version_to_record(version: PaperVersion) -> PaperVersionRecord:
    return PaperVersionRecord(
        paper_version_id=version.paper_version_id,
        paper_id=version.paper_id,
        version=version.version,
        source_version=version.source_version,
        checksum=version.checksum,
        parser_version=version.parser_version,
        storage_path=version.storage_path,
        downloaded_at=version.downloaded_at,
        file_size_bytes=version.file_size_bytes,
        parsed_artifact_path=version.parsed_artifact_path,
        parsed_at=version.parsed_at,
        parser_name=version.parser_name,
        page_count=version.page_count,
        section_count=version.section_count,
        warning_count=version.warning_count,
        chunked_artifact_path=version.chunked_artifact_path,
        chunked_at=version.chunked_at,
        chunk_count=version.chunk_count,
        chunking_version=version.chunking_version,
        chunk_artifact_checksum=version.chunk_artifact_checksum,
        chunk_config_fingerprint=version.chunk_config_fingerprint,
        vector_indexed_at=version.vector_indexed_at,
        vector_count=version.vector_count,
        embedding_provider=version.embedding_provider,
        embedding_model=version.embedding_model,
        embedding_config_fingerprint=version.embedding_config_fingerprint,
        vector_generation_fingerprint=version.vector_generation_fingerprint,
        qdrant_collection=version.qdrant_collection,
        graph_extraction_artifact_path=version.graph_extraction_artifact_path,
        graph_extracted_at=version.graph_extracted_at,
        entity_count=version.entity_count,
        relationship_count=version.relationship_count,
        extraction_version=version.extraction_version,
        extraction_config_fingerprint=version.extraction_config_fingerprint,
        graph_extraction_generation_fingerprint=version.graph_extraction_generation_fingerprint,
        graph_extraction_artifact_checksum=version.graph_extraction_artifact_checksum,
        graph_indexed_at=version.graph_indexed_at,
        canonical_entity_count=version.canonical_entity_count,
        graph_relationship_count=version.graph_relationship_count,
        canonicalization_config_fingerprint=version.canonicalization_config_fingerprint,
        graph_index_generation_fingerprint=version.graph_index_generation_fingerprint,
        neo4j_database=version.neo4j_database,
    )


# --- Ingestion job -------------------------------------------------------


def ingestion_job_record_to_domain(record: IngestionJobRecord) -> IngestionJob:
    return IngestionJob(
        ingestion_job_id=record.ingestion_job_id,
        paper_id=record.paper_id,
        paper_version_id=record.paper_version_id,
        status=IngestionStatus(record.status),
        failed_stage=IngestionStatus(record.failed_stage) if record.failed_stage else None,
        failure_reason=record.failure_reason,
        retry_count=record.retry_count,
        pipeline_version=record.pipeline_version,
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        updated_at=record.updated_at,
    )


def domain_ingestion_job_to_record(job: IngestionJob) -> IngestionJobRecord:
    return IngestionJobRecord(
        ingestion_job_id=job.ingestion_job_id,
        paper_id=job.paper_id,
        paper_version_id=job.paper_version_id,
        status=job.status.value,
        failed_stage=job.failed_stage.value if job.failed_stage else None,
        failure_reason=job.failure_reason,
        retry_count=job.retry_count,
        pipeline_version=job.pipeline_version,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


# --- Ingestion step -------------------------------------------------------


def step_record_to_domain(record: IngestionStepRecord) -> IngestionStepState:
    return IngestionStepState(
        step_id=record.step_id,
        ingestion_job_id=record.ingestion_job_id,
        stage=ProcessingStage(record.stage),
        status=StepStatus(record.status),
        attempt=record.attempt,
        started_at=record.started_at,
        completed_at=record.completed_at,
        error_code=record.error_code,
        error_message=record.error_message,
        metadata=dict(record.step_metadata or {}),
    )


def domain_step_to_record(step: IngestionStepState) -> IngestionStepRecord:
    return IngestionStepRecord(
        ingestion_job_id=step.ingestion_job_id,
        stage=step.stage.value,
        status=step.status.value,
        attempt=step.attempt,
        started_at=step.started_at,
        completed_at=step.completed_at,
        error_code=step.error_code,
        error_message=step.error_message,
        step_metadata=dict(step.metadata),
    )


def update_step_record_from_domain(record: IngestionStepRecord, step: IngestionStepState) -> None:
    record.status = step.status.value
    record.started_at = step.started_at
    record.completed_at = step.completed_at
    record.error_code = step.error_code
    record.error_message = step.error_message
    record.step_metadata = dict(step.metadata)
