"""Scientific PDF parsing service: orchestrates state, parser invocation,
and idempotency/reconciliation for the parsing stage.

Target flow (prompt #2)::

    DOWNLOADED -> validate raw artifact -> PARSING -> ScientificPaperParser
    -> section recovery -> persist parsed.json -> persist parser metadata
    -> PARSED

Never continues into chunking.

Reconciliation mirrors Prompt 4's filesystem-first design: `parsed.json`
on disk (re-verified against the *current* PDF checksum and this parser's
name/version) is the source of truth for "has this already been parsed",
not whatever PostgreSQL currently records.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from app.core.exceptions import InvalidPdfError
from app.domain.enums import IngestionStatus, ProcessingStage, StepStatus
from app.domain.ingestion import IngestionJob, IngestionStepState
from app.domain.papers import Paper, PaperVersion
from app.ingestion.checksums import sha256_file
from app.ingestion.download.storage import PaperStorage
from app.ingestion.paper_resolution import resolve_paper, resolve_version
from app.ingestion.parsing.models import ParsedPaperDocument
from app.ingestion.parsing.parser import ScientificPaperParser
from app.ingestion.parsing.storage import ParsedArtifactStorage
from app.storage.postgres.repositories.ingestion import IngestionRepository
from app.storage.postgres.repositories.papers import PaperRepository

logger = logging.getLogger(__name__)


class ParseResult(BaseModel):
    """Outcome of `PaperParsingService.parse()`."""

    model_config = {"arbitrary_types_allowed": True}

    job: IngestionJob
    document: ParsedPaperDocument
    parse_reused: bool = False


class PaperParsingService:
    """Explicit PDF parsing: the only thing FastAPI routes should call for
    `POST /api/v1/papers/{paper_id}/parse`."""

    def __init__(
        self,
        parser: ScientificPaperParser,
        pdf_storage: PaperStorage,
        parsed_storage: ParsedArtifactStorage,
        paper_repository: PaperRepository,
        ingestion_repository: IngestionRepository,
    ) -> None:
        self._parser = parser
        self._pdf_storage = pdf_storage
        self._parsed_storage = parsed_storage
        self._paper_repo = paper_repository
        self._ingestion_repo = ingestion_repository

    def parse(self, paper_id: str, paper_version_id: str | None = None) -> ParseResult:
        paper = resolve_paper(self._paper_repo, paper_id)
        version = resolve_version(self._paper_repo, paper, paper_version_id)
        self._validate_ready_for_parsing(paper, version)

        job = self._ingestion_repo.create_ingestion_job(
            paper_id=paper.paper_id, paper_version_id=version.paper_version_id
        )

        result, job = self._reconcile_existing_parse(paper, version, job)
        if result is not None:
            return result

        return self._parse_and_finalize(paper, version, job)

    # --- Validation -----------------------------------------------------

    def _validate_ready_for_parsing(self, paper: Paper, version: PaperVersion) -> Path:
        """Preconditions (prompt #7): the raw artifact exists, and -- where
        checksum information is available -- still matches what was
        recorded when it was downloaded. Whether the *ingestion state*
        permits parsing is enforced separately, by the state machine
        itself when a transition to PARSING is actually attempted."""

        pdf_path = self._pdf_storage.get_path(
            source=paper.source, source_id=paper.source_id, version=version.version
        )
        if not pdf_path.is_file():
            raise InvalidPdfError(
                f"no raw PDF artifact found for {version.paper_version_id}; ingest it first"
            )
        if version.checksum:
            actual_checksum = sha256_file(pdf_path)
            if actual_checksum != version.checksum:
                raise InvalidPdfError(
                    f"stored PDF for {version.paper_version_id} failed checksum re-verification"
                )
        return pdf_path

    # --- Reconciliation ---------------------------------------------------

    def _reconcile_existing_parse(
        self, paper: Paper, version: PaperVersion, job: IngestionJob
    ) -> tuple[ParseResult | None, IngestionJob]:
        """Filesystem-first reconciliation, mirroring Prompt 4: check
        whether a valid `parsed.json` already exists for the *current*
        parser name/version before deciding whether to actually reparse.

        Returns `(result, job)`, same contract as Prompt 4's acquisition
        reconciliation: `result` is not `None` when an existing parse was
        reused; otherwise `job` is what the caller should parse into.
        """

        existing = self._parsed_storage.try_read(
            source=paper.source, source_id=paper.source_id, version=version.version
        )
        if existing is None:
            return None, self._recreate_job_if_stale_parse(
                paper, version, job, reason="parsed artifact missing or corrupt"
            )

        stale_reason = self._staleness_reason(existing, version)
        if stale_reason is not None:
            logger.info(
                "existing parsed artifact is stale, reparsing paper_id=%s "
                "paper_version_id=%s reason=%r status=stale",
                paper.paper_id,
                version.paper_version_id,
                stale_reason,
            )
            return None, self._recreate_job_if_stale_parse(paper, version, job, reason=stale_reason)

        # Valid, current parse exists -- reconcile PostgreSQL to match it
        # (covers "parsed.json was finalized but the final DB write
        # failed") without reparsing.
        was_already_recorded = version.parsed_artifact_path is not None
        parsed_path = self._parsed_storage.get_path(
            source=paper.source, source_id=paper.source_id, version=version.version
        )
        self._paper_repo.update_version_parse_result(
            version.paper_version_id,
            parsed_artifact_path=str(parsed_path),
            parsed_at=version.parsed_at or datetime.now(timezone.utc),
            parser_name=existing.parser_name,
            parser_version=existing.parser_version,
            page_count=existing.page_count,
            section_count=len(existing.sections),
            warning_count=len(existing.warnings),
        )
        job = self._advance_job_to_parsed(job)

        if not was_already_recorded:
            now = datetime.now(timezone.utc)
            self._record_step(
                job.ingestion_job_id,
                self._next_attempt_number(job.ingestion_job_id),
                StepStatus.COMPLETED,
                started_at=now,
                completed_at=now,
                metadata={
                    "page_count": existing.page_count,
                    "section_count": len(existing.sections),
                    "reconciled": True,
                },
            )

        logger.info(
            "reused existing valid parse paper_id=%s paper_version_id=%s "
            "ingestion_job_id=%s status=ok",
            paper.paper_id,
            version.paper_version_id,
            job.ingestion_job_id,
        )
        return ParseResult(job=job, document=existing, parse_reused=True), job

    def _staleness_reason(
        self, existing: ParsedPaperDocument, version: PaperVersion
    ) -> str | None:
        """Reparse invalidation triggers (prompt #29): a different parser
        produced the existing artifact, or the underlying PDF changed
        since it was parsed. Calling the endpoint twice with nothing
        changed is NOT a reason to reparse."""

        if existing.parser_name != self._parser.name or existing.parser_version != self._parser.version:
            return (
                f"parser changed ({existing.parser_name}/{existing.parser_version} -> "
                f"{self._parser.name}/{self._parser.version})"
            )
        if version.checksum and existing.source_pdf_checksum and existing.source_pdf_checksum != version.checksum:
            return "source PDF checksum changed since this parse was produced"
        return None

    def _recreate_job_if_stale_parse(
        self, paper: Paper, version: PaperVersion, job: IngestionJob, *, reason: str
    ) -> IngestionJob:
        """If `job` had already reached PARSED but the artifact backing
        that claim is gone/corrupt/stale, mark it failed and start a fresh
        job -- then fast-forward that fresh job straight to DOWNLOADED,
        since `_validate_ready_for_parsing` already confirmed the PDF
        itself is still valid; only the parse needs redoing, not a
        re-download."""

        if job.status != IngestionStatus.PARSED:
            return job

        self._ingestion_repo.mark_job_failed(
            job.ingestion_job_id, failed_stage=IngestionStatus.PARSED, failure_reason=reason
        )
        logger.warning(
            "marked stale parse job failed and starting a new one paper_id=%s "
            "paper_version_id=%s ingestion_job_id=%s reason=%r status=stale",
            paper.paper_id,
            version.paper_version_id,
            job.ingestion_job_id,
            reason,
        )
        new_job = self._ingestion_repo.create_ingestion_job(
            paper_id=paper.paper_id, paper_version_id=version.paper_version_id
        )
        return self._ensure_job_at_downloaded(new_job)

    def _ensure_job_at_downloaded(self, job: IngestionJob) -> IngestionJob:
        if job.status == IngestionStatus.DISCOVERED:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.DOWNLOADING
            )
        if job.status == IngestionStatus.DOWNLOADING:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.DOWNLOADED
            )
        return job

    def _advance_job_to_parsed(self, job: IngestionJob) -> IngestionJob:
        job = self._ensure_job_at_downloaded(job)
        if job.status == IngestionStatus.DOWNLOADED:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.PARSING
            )
        if job.status == IngestionStatus.PARSING:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.PARSED
            )
        return job

    # --- Parsing ---------------------------------------------------------

    def _parse_and_finalize(
        self, paper: Paper, version: PaperVersion, job: IngestionJob
    ) -> ParseResult:
        started_monotonic = time.monotonic()
        started_at = datetime.now(timezone.utc)
        attempt = self._next_attempt_number(job.ingestion_job_id)

        pdf_path = self._pdf_storage.get_path(
            source=paper.source, source_id=paper.source_id, version=version.version
        )

        job = self._ensure_job_at_downloaded(job)
        if job.status == IngestionStatus.DOWNLOADED:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.PARSING
            )

        try:
            document = self._parser.parse(
                pdf_path, paper_id=paper.paper_id, paper_version_id=version.paper_version_id
            )
            document = document.model_copy(update={"source_pdf_checksum": version.checksum or ""})
            parsed_path = self._parsed_storage.write(
                document, source=paper.source, source_id=paper.source_id, version=version.version
            )
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
                failed_stage=IngestionStatus.PARSING,
                failure_reason=str(exc)[:2000],
            )
            logger.warning(
                "pdf parsing failed paper_id=%s paper_version_id=%s ingestion_job_id=%s "
                "attempt=%d duration_ms=%d status=error error=%s",
                paper.paper_id,
                version.paper_version_id,
                job.ingestion_job_id,
                attempt,
                duration_ms,
                exc,
            )
            raise

        self._paper_repo.update_version_parse_result(
            version.paper_version_id,
            parsed_artifact_path=str(parsed_path),
            parsed_at=datetime.now(timezone.utc),
            parser_name=document.parser_name,
            parser_version=document.parser_version,
            page_count=document.page_count,
            section_count=len(document.sections),
            warning_count=len(document.warnings),
        )
        job = self._ingestion_repo.transition_job_status(
            job.ingestion_job_id, IngestionStatus.PARSED
        )

        duration_ms = int((time.monotonic() - started_monotonic) * 1000)
        self._record_step(
            job.ingestion_job_id,
            attempt,
            StepStatus.COMPLETED,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            metadata={
                "page_count": document.page_count,
                "section_count": len(document.sections),
                "warning_count": len(document.warnings),
                "duration_ms": duration_ms,
            },
        )
        logger.info(
            "pdf parsing succeeded paper_id=%s paper_version_id=%s ingestion_job_id=%s "
            "parser_name=%s parser_version=%s page_count=%d section_count=%d "
            "warning_count=%d duration_ms=%d status=ok",
            paper.paper_id,
            version.paper_version_id,
            job.ingestion_job_id,
            document.parser_name,
            document.parser_version,
            document.page_count,
            len(document.sections),
            len(document.warnings),
            duration_ms,
        )
        return ParseResult(job=job, document=document, parse_reused=False)

    # --- Steps -------------------------------------------------------------

    def _next_attempt_number(self, ingestion_job_id: str) -> int:
        existing = [
            step
            for step in self._ingestion_repo.list_steps(ingestion_job_id)
            if step.stage == ProcessingStage.PARSE
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
                stage=ProcessingStage.PARSE,
                status=status,
                attempt=attempt,
                started_at=started_at,
                completed_at=completed_at,
                error_code=error_code,
                error_message=error_message,
                metadata=metadata or {},
            )
        )
