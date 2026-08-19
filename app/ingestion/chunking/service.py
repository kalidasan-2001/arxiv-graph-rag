"""Section-aware chunking service: orchestrates state, chunker invocation,
and idempotency/reconciliation for the chunking stage.

Target flow (prompt #2)::

    PARSED -> validate parsed artifact -> CHUNKING -> ScientificChunker
    -> section-aware splitting -> validate chunks -> atomic chunks.json
    -> persist chunk metadata -> CHUNKED

Never continues into embedding.

Reconciliation mirrors Prompt 5's filesystem-first design: `chunks.json`
on disk (re-verified against the *current* parsed-artifact checksum and
the current chunking configuration) is the source of truth for "has this
already been chunked", not whatever PostgreSQL currently records.
"""

import logging
import time
from datetime import datetime, timezone

from pydantic import BaseModel

from app.core.exceptions import ParseArtifactNotFoundError
from app.domain.enums import IngestionStatus, ProcessingStage, StepStatus
from app.domain.ingestion import IngestionJob, IngestionStepState
from app.domain.papers import Paper, PaperVersion
from app.ingestion.checksums import sha256_file
from app.ingestion.chunking.chunker import ScientificChunker
from app.ingestion.chunking.models import ChunkedPaperDocument
from app.ingestion.chunking.storage import ChunkArtifactStorage
from app.ingestion.paper_resolution import resolve_paper, resolve_version
from app.ingestion.parsing.models import ParsedPaperDocument
from app.ingestion.parsing.storage import ParsedArtifactStorage
from app.storage.postgres.repositories.ingestion import IngestionRepository
from app.storage.postgres.repositories.papers import PaperRepository

logger = logging.getLogger(__name__)

# Every status a job can hold *at or after* CHUNKED (prompt 7 made
# VECTOR_INDEXING/VECTOR_INDEXED reachable; prompt 8 will add
# GRAPH_INDEXING/GRAPH_INDEXED) -- any of these means chunking has already
# happened at least once for this job and a legitimate rechunk must mark it
# failed and start fresh, not attempt an illegal direct transition back to
# CHUNKED. Discovered via a real integration test: re-chunking a paper that
# had already reached VECTOR_INDEXED previously raised
# `InvalidIngestionTransitionError` because this reconciliation helper only
# ever checked `== CHUNKED`, the sole status reachable when it was written.
_STATUSES_AT_OR_PAST_CHUNKED = frozenset(
    {
        IngestionStatus.CHUNKED,
        IngestionStatus.VECTOR_INDEXING,
        IngestionStatus.VECTOR_INDEXED,
        IngestionStatus.GRAPH_INDEXING,
        IngestionStatus.GRAPH_INDEXED,
    }
)


class ChunkResult(BaseModel):
    """Outcome of `ChunkingService.chunk()`."""

    job: IngestionJob
    document: ChunkedPaperDocument
    chunk_reused: bool = False


class ChunkingService:
    """Explicit chunking: the only thing FastAPI routes should call for
    `POST /api/v1/papers/{paper_id}/chunk`."""

    def __init__(
        self,
        chunker: ScientificChunker,
        parsed_storage: ParsedArtifactStorage,
        chunk_storage: ChunkArtifactStorage,
        paper_repository: PaperRepository,
        ingestion_repository: IngestionRepository,
    ) -> None:
        self._chunker = chunker
        self._parsed_storage = parsed_storage
        self._chunk_storage = chunk_storage
        self._paper_repo = paper_repository
        self._ingestion_repo = ingestion_repository

    def chunk(self, paper_id: str, paper_version_id: str | None = None) -> ChunkResult:
        paper = resolve_paper(self._paper_repo, paper_id)
        version = resolve_version(self._paper_repo, paper, paper_version_id)
        parsed_document, parsed_checksum = self._validate_ready_for_chunking(paper, version)

        job = self._ingestion_repo.create_ingestion_job(
            paper_id=paper.paper_id, paper_version_id=version.paper_version_id
        )

        result, job = self._reconcile_existing_chunks(
            paper, version, job, parsed_checksum=parsed_checksum
        )
        if result is not None:
            return result

        return self._chunk_and_finalize(
            paper, version, job, parsed_document=parsed_document, parsed_checksum=parsed_checksum
        )

    # --- Validation -----------------------------------------------------

    def _validate_ready_for_chunking(
        self, paper: Paper, version: PaperVersion
    ) -> tuple[ParsedPaperDocument, str]:
        """Chunking never reads `paper.pdf` directly (prompt #31) -- only
        the structured parsed artifact. A missing/corrupt `parsed.json` is
        not a reconciliation case here: without it there is nothing to
        chunk from, or to validate an existing `chunks.json` against."""

        document = self._parsed_storage.try_read(
            source=paper.source, source_id=paper.source_id, version=version.version
        )
        if document is None:
            raise ParseArtifactNotFoundError(
                f"no valid parsed artifact for {version.paper_version_id}; parse it first"
            )
        parsed_path = self._parsed_storage.get_path(
            source=paper.source, source_id=paper.source_id, version=version.version
        )
        return document, sha256_file(parsed_path)

    # --- Reconciliation ---------------------------------------------------

    def _reconcile_existing_chunks(
        self, paper: Paper, version: PaperVersion, job: IngestionJob, *, parsed_checksum: str
    ) -> tuple[ChunkResult | None, IngestionJob]:
        """Filesystem-first reconciliation, mirroring Prompt 5."""

        existing = self._chunk_storage.try_read(
            source=paper.source, source_id=paper.source_id, version=version.version
        )
        if existing is None:
            return None, self._recreate_job_if_stale_chunks(
                paper, version, job, reason="chunk artifact missing or corrupt"
            )

        stale_reason = self._staleness_reason(existing, parsed_checksum=parsed_checksum)
        if stale_reason is not None:
            logger.info(
                "existing chunk artifact is stale, rechunking paper_id=%s "
                "paper_version_id=%s reason=%r status=stale",
                paper.paper_id,
                version.paper_version_id,
                stale_reason,
            )
            return None, self._recreate_job_if_stale_chunks(paper, version, job, reason=stale_reason)

        # Valid, current chunk artifact exists -- reconcile PostgreSQL to
        # match it (covers "chunks.json was finalized but the final DB
        # write failed") without rechunking.
        was_already_recorded = version.chunked_artifact_path is not None
        chunk_path = self._chunk_storage.get_path(
            source=paper.source, source_id=paper.source_id, version=version.version
        )
        self._paper_repo.update_version_chunk_result(
            version.paper_version_id,
            chunked_artifact_path=str(chunk_path),
            chunked_at=version.chunked_at or datetime.now(timezone.utc),
            chunk_count=len(existing.chunks),
            chunking_version=existing.chunking.version,
            chunk_artifact_checksum=sha256_file(chunk_path),
            chunk_config_fingerprint=existing.chunking.config_fingerprint,
        )
        job = self._advance_job_to_chunked(job)

        if not was_already_recorded:
            now = datetime.now(timezone.utc)
            self._record_step(
                job.ingestion_job_id,
                self._next_attempt_number(job.ingestion_job_id),
                StepStatus.COMPLETED,
                started_at=now,
                completed_at=now,
                metadata={"chunk_count": len(existing.chunks), "reconciled": True},
            )

        logger.info(
            "reused existing valid chunks paper_id=%s paper_version_id=%s "
            "ingestion_job_id=%s status=ok",
            paper.paper_id,
            version.paper_version_id,
            job.ingestion_job_id,
        )
        return ChunkResult(job=job, document=existing, chunk_reused=True), job

    def _staleness_reason(self, existing: ChunkedPaperDocument, *, parsed_checksum: str) -> str | None:
        """Reparse invalidation triggers (prompt #29, hardened by prompt
        6.1): the parsed source changed, or the *effective* chunking
        configuration changed. Compares `config_fingerprint`, not
        `chunking.version` alone -- a bare version string is a
        human-maintained label nothing forces a developer to bump when
        e.g. `CHUNK_SIZE_TOKENS` changes, so it alone is not sufficient
        proof that an existing artifact is still valid. Calling the
        endpoint twice with nothing changed is NOT a reason to rechunk."""

        if existing.parsed_artifact_checksum != parsed_checksum:
            return "parsed artifact checksum changed since these chunks were produced"
        if existing.chunking.config_fingerprint != self._chunker.config_fingerprint:
            return (
                "chunking configuration changed "
                f"({existing.chunking.config_fingerprint} -> {self._chunker.config_fingerprint})"
            )
        return None

    def _recreate_job_if_stale_chunks(
        self, paper: Paper, version: PaperVersion, job: IngestionJob, *, reason: str
    ) -> IngestionJob:
        """If `job` had already reached CHUNKED (or any later stage -- a
        paper can legitimately be rechunked after vector/graph indexing,
        which must then invalidate those downstream artifacts, not crash)
        but the chunk artifact backing that claim is gone/corrupt/stale,
        mark it failed and start a fresh job -- then fast-forward it to
        PARSED, since the parsed source was already re-verified valid;
        only chunking needs redoing."""

        if job.status not in _STATUSES_AT_OR_PAST_CHUNKED:
            return job

        self._ingestion_repo.mark_job_failed(
            job.ingestion_job_id, failed_stage=job.status, failure_reason=reason
        )
        logger.warning(
            "marked stale chunk job failed and starting a new one paper_id=%s "
            "paper_version_id=%s ingestion_job_id=%s reason=%r status=stale",
            paper.paper_id,
            version.paper_version_id,
            job.ingestion_job_id,
            reason,
        )
        new_job = self._ingestion_repo.create_ingestion_job(
            paper_id=paper.paper_id, paper_version_id=version.paper_version_id
        )
        return self._ensure_job_at_parsed(new_job)

    def _ensure_job_at_parsed(self, job: IngestionJob) -> IngestionJob:
        if job.status == IngestionStatus.DISCOVERED:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.DOWNLOADING
            )
        if job.status == IngestionStatus.DOWNLOADING:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.DOWNLOADED
            )
        if job.status == IngestionStatus.DOWNLOADED:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.PARSING
            )
        if job.status == IngestionStatus.PARSING:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.PARSED
            )
        return job

    def _advance_job_to_chunked(self, job: IngestionJob) -> IngestionJob:
        """No-op if `job` is already at CHUNKED or a later stage (prompt 7
        made that reachable): reusing still-valid chunks on a paper that
        has already progressed further (e.g. to VECTOR_INDEXED) must never
        regress its reported status back to CHUNKED -- nothing about
        vector indexing was invalidated, so nothing should look undone."""

        if job.status in _STATUSES_AT_OR_PAST_CHUNKED:
            return job

        job = self._ensure_job_at_parsed(job)
        if job.status == IngestionStatus.PARSED:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.CHUNKING
            )
        if job.status == IngestionStatus.CHUNKING:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.CHUNKED
            )
        return job

    # --- Chunking ---------------------------------------------------------

    def _chunk_and_finalize(
        self,
        paper: Paper,
        version: PaperVersion,
        job: IngestionJob,
        *,
        parsed_document: ParsedPaperDocument,
        parsed_checksum: str,
    ) -> ChunkResult:
        started_monotonic = time.monotonic()
        started_at = datetime.now(timezone.utc)
        attempt = self._next_attempt_number(job.ingestion_job_id)

        job = self._ensure_job_at_parsed(job)
        if job.status == IngestionStatus.PARSED:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.CHUNKING
            )

        try:
            document = self._chunker.chunk(parsed_document)
            document = document.model_copy(update={"parsed_artifact_checksum": parsed_checksum})
            chunk_path = self._chunk_storage.write(
                document, source=paper.source, source_id=paper.source_id, version=version.version
            )
            chunk_artifact_checksum = sha256_file(chunk_path)
        except Exception as exc:
            duration_ms = int((time.monotonic() - started_monotonic) * 1000)
            self._record_step(
                job.ingestion_job_id,
                attempt,
                StepStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                error_code=type(exc).__name__,
                error_message=str(exc)[:500],
                metadata={"duration_ms": duration_ms},
            )
            self._ingestion_repo.mark_job_failed(
                job.ingestion_job_id,
                failed_stage=IngestionStatus.CHUNKING,
                failure_reason=str(exc)[:2000],
            )
            logger.warning(
                "chunking failed paper_id=%s paper_version_id=%s ingestion_job_id=%s "
                "attempt=%d duration_ms=%d status=error error=%s",
                paper.paper_id,
                version.paper_version_id,
                job.ingestion_job_id,
                attempt,
                duration_ms,
                exc,
            )
            raise

        self._paper_repo.update_version_chunk_result(
            version.paper_version_id,
            chunked_artifact_path=str(chunk_path),
            chunked_at=datetime.now(timezone.utc),
            chunk_count=len(document.chunks),
            chunking_version=document.chunking.version,
            chunk_artifact_checksum=chunk_artifact_checksum,
            chunk_config_fingerprint=document.chunking.config_fingerprint,
        )
        job = self._ingestion_repo.transition_job_status(
            job.ingestion_job_id, IngestionStatus.CHUNKED
        )

        duration_ms = int((time.monotonic() - started_monotonic) * 1000)
        average_tokens = document.diagnostics.average_tokens
        self._record_step(
            job.ingestion_job_id,
            attempt,
            StepStatus.COMPLETED,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            metadata={
                "chunk_count": len(document.chunks),
                "average_tokens": average_tokens,
                "duration_ms": duration_ms,
            },
        )
        logger.info(
            "chunking succeeded paper_id=%s paper_version_id=%s ingestion_job_id=%s "
            "chunking_version=%s chunk_size=%d overlap=%d chunk_count=%d "
            "average_tokens=%.1f duration_ms=%d artifact_reused=False status=ok",
            paper.paper_id,
            version.paper_version_id,
            job.ingestion_job_id,
            document.chunking.version,
            document.chunking.chunk_size_tokens,
            document.chunking.chunk_overlap_tokens,
            len(document.chunks),
            average_tokens,
            duration_ms,
        )
        return ChunkResult(job=job, document=document, chunk_reused=False)

    # --- Steps -------------------------------------------------------------

    def _next_attempt_number(self, ingestion_job_id: str) -> int:
        existing = [
            step
            for step in self._ingestion_repo.list_steps(ingestion_job_id)
            if step.stage == ProcessingStage.CHUNK
        ]
        return max((step.attempt for step in existing), default=0) + 1

    def _record_step(
        self,
        ingestion_job_id: str,
        attempt: int,
        status: StepStatus,
        *,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._ingestion_repo.record_step(
            IngestionStepState(
                ingestion_job_id=ingestion_job_id,
                stage=ProcessingStage.CHUNK,
                status=status,
                attempt=attempt,
                started_at=started_at,
                completed_at=completed_at,
                error_code=error_code,
                error_message=error_message,
                metadata=metadata or {},
            )
        )
