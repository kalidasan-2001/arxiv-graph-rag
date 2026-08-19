"""PDF acquisition service: orchestrates state, download, checksum, and idempotency.

Target flow (prompt #2)::

    DISCOVERED -> DOWNLOADING -> download PDF -> validate -> SHA-256
    -> persist raw PDF -> persist checksum/path metadata -> DOWNLOADED

Never continues into parsing -- CLAUDE.md's PARSE stage is a later prompt,
and this service never calls `transition_job_status(..., PARSING)`.

Reconciliation is filesystem-first, not database-first (prompt #27/#28):
every call checks whether a valid artifact already exists at the
deterministic storage path *regardless of what PostgreSQL currently
believes*, because the one scenario this stage must handle safely is "the
file was finalized but the final DB write failed" -- in that scenario the
database alone would not know the artifact exists, but the filesystem
does.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel

from app.core.config import Settings
from app.core.exceptions import InvalidPdfError, PdfNotFoundError, UnsafePdfUrlError
from app.domain.enums import IngestionStatus, ProcessingStage, StepStatus
from app.domain.ingestion import IngestionJob, IngestionStepState
from app.domain.papers import Paper, PaperVersion
from app.ingestion.checksums import sha256_file
from app.ingestion.download.client import PdfDownloadClient
from app.ingestion.download.storage import PaperStorage
from app.ingestion.paper_resolution import resolve_paper, resolve_version
from app.storage.postgres.repositories.ingestion import IngestionRepository
from app.storage.postgres.repositories.papers import PaperRepository

logger = logging.getLogger(__name__)

_PDF_SIGNATURE = b"%PDF-"
# For this initial arXiv-only implementation (prompt #18): only these
# hosts are trusted. A future source provider implements its own
# allowlist rather than this one growing ad hoc.
_TRUSTED_PDF_HOSTS = frozenset({"arxiv.org", "export.arxiv.org", "www.arxiv.org"})


class AcquisitionResult(BaseModel):
    """Outcome of `PdfAcquisitionService.ingest()`."""

    job: IngestionJob
    artifact_reused: bool = False


class PdfAcquisitionService:
    """Explicit PDF ingestion: the only thing FastAPI routes should call
    for `POST /api/v1/papers/{paper_id}/ingest`."""

    def __init__(
        self,
        settings: Settings,
        client: PdfDownloadClient,
        storage: PaperStorage,
        paper_repository: PaperRepository,
        ingestion_repository: IngestionRepository,
    ) -> None:
        self._settings = settings
        self._client = client
        self._storage = storage
        self._paper_repo = paper_repository
        self._ingestion_repo = ingestion_repository

    def ingest(self, paper_id: str, paper_version_id: str | None = None) -> AcquisitionResult:
        paper = resolve_paper(self._paper_repo, paper_id)
        version = resolve_version(self._paper_repo, paper, paper_version_id)
        self._validate_pdf_url(paper.pdf_url, paper_id=paper.paper_id)

        job = self._ingestion_repo.create_ingestion_job(
            paper_id=paper.paper_id, paper_version_id=version.paper_version_id
        )

        result, job = self._reconcile_existing_artifact(paper, version, job)
        if result is not None:
            return result

        return self._download_and_finalize(paper, version, job)

    # --- Validation -----------------------------------------------------

    def _validate_pdf_url(self, pdf_url: str | None, *, paper_id: str) -> None:
        """SSRF prevention (prompt #17/#18): only a trusted, already-
        discovered arXiv PDF URL may ever be fetched -- never an arbitrary
        caller-supplied URL, and never an unexpected host."""

        if not pdf_url:
            raise PdfNotFoundError(f"paper {paper_id} has no known pdf_url")

        parsed = urlsplit(pdf_url)
        if parsed.scheme not in ("http", "https"):
            raise UnsafePdfUrlError(f"unsupported URL scheme for PDF download: {parsed.scheme!r}")

        host = (parsed.hostname or "").lower()
        if host not in _TRUSTED_PDF_HOSTS:
            logger.warning(
                "rejected pdf_url with untrusted host paper_id=%s host=%s status=rejected",
                paper_id,
                host,
            )
            raise UnsafePdfUrlError(f"PDF URL host is not in the trusted allowlist: {host!r}")

    # --- Reconciliation ---------------------------------------------------

    def _reconcile_existing_artifact(
        self, paper: Paper, version: PaperVersion, job: IngestionJob
    ) -> tuple[AcquisitionResult | None, IngestionJob]:
        """Check the filesystem for an already-valid artifact and reuse it.

        Returns `(result, job)`: `result` is not `None` when an existing
        artifact was successfully reused (job state reconciled as needed);
        when `result` is `None`, `job` is the job the caller should
        actually download into (possibly a freshly created one, if the
        previous job had to be marked failed/stale first).
        """

        final_path = self._storage.get_path(
            source=paper.source, source_id=paper.source_id, version=version.version
        )

        if not final_path.is_file():
            return None, self._recreate_job_if_stale(
                paper, version, job, reason="recorded artifact is missing from local storage"
            )

        actual_checksum = sha256_file(final_path)
        if version.checksum is not None and actual_checksum != version.checksum:
            logger.warning(
                "existing artifact failed checksum verification paper_id=%s "
                "paper_version_id=%s path=%s status=corrupt",
                paper.paper_id,
                version.paper_version_id,
                final_path,
            )
            self._storage.delete(
                source=paper.source, source_id=paper.source_id, version=version.version
            )
            return None, self._recreate_job_if_stale(
                paper, version, job, reason="stored artifact failed checksum verification"
            )

        # Artifact on disk is valid. Reconcile PostgreSQL to match it --
        # this also covers the "final DB write failed after the file was
        # finalized" case, where `version.checksum` would still be `None`.
        was_already_recorded = version.checksum is not None
        file_size = final_path.stat().st_size
        self._paper_repo.update_version_artifact(
            version.paper_version_id,
            checksum=actual_checksum,
            storage_path=str(final_path),
            file_size_bytes=file_size,
            downloaded_at=version.downloaded_at or datetime.now(timezone.utc),
        )
        job = self._advance_job_to_downloaded(job)

        if not was_already_recorded:
            now = datetime.now(timezone.utc)
            self._record_step(
                job.ingestion_job_id,
                self._next_attempt_number(job.ingestion_job_id),
                StepStatus.COMPLETED,
                started_at=now,
                completed_at=now,
                metadata={
                    "bytes_downloaded": file_size,
                    "checksum": actual_checksum,
                    "reconciled": True,
                },
            )

        logger.info(
            "reused existing valid artifact paper_id=%s paper_version_id=%s "
            "ingestion_job_id=%s status=ok",
            paper.paper_id,
            version.paper_version_id,
            job.ingestion_job_id,
        )
        return AcquisitionResult(job=job, artifact_reused=True), job

    def _recreate_job_if_stale(
        self, paper: Paper, version: PaperVersion, job: IngestionJob, *, reason: str
    ) -> IngestionJob:
        """If `job` had already reached DOWNLOADED but the artifact backing
        that claim is gone/corrupt, mark it failed and start a fresh job.

        Only DOWNLOADED needs this: a job still at DISCOVERED/DOWNLOADING
        can simply retry its own download attempt (handled by the normal
        download path), since it never claimed success in the first place.
        """

        if job.status != IngestionStatus.DOWNLOADED:
            return job

        self._ingestion_repo.mark_job_failed(
            job.ingestion_job_id,
            failed_stage=IngestionStatus.DOWNLOADED,
            failure_reason=reason,
        )
        logger.warning(
            "marked stale job failed and starting a new one paper_id=%s "
            "paper_version_id=%s ingestion_job_id=%s reason=%r status=stale",
            paper.paper_id,
            version.paper_version_id,
            job.ingestion_job_id,
            reason,
        )
        return self._ingestion_repo.create_ingestion_job(
            paper_id=paper.paper_id, paper_version_id=version.paper_version_id
        )

    def _advance_job_to_downloaded(self, job: IngestionJob) -> IngestionJob:
        if job.status == IngestionStatus.DISCOVERED:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.DOWNLOADING
            )
        if job.status == IngestionStatus.DOWNLOADING:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.DOWNLOADED
            )
        return job

    # --- Download ---------------------------------------------------------

    def _download_and_finalize(
        self, paper: Paper, version: PaperVersion, job: IngestionJob
    ) -> AcquisitionResult:
        started_monotonic = time.monotonic()
        started_at = datetime.now(timezone.utc)
        attempt = self._next_attempt_number(job.ingestion_job_id)
        temp_path = self._storage.get_temp_path(
            source=paper.source, source_id=paper.source_id, version=version.version
        )

        if job.status == IngestionStatus.DISCOVERED:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.DOWNLOADING
            )

        try:
            download_result = self._client.download(paper.pdf_url, temp_path)
            _validate_pdf_signature(temp_path)
            checksum = sha256_file(temp_path)
            final_path = self._storage.finalize(
                temp_path, source=paper.source, source_id=paper.source_id, version=version.version
            )
        except Exception as exc:
            self._storage.cleanup_temp(temp_path)
            duration_ms = int((time.monotonic() - started_monotonic) * 1000)
            self._record_step(
                job.ingestion_job_id,
                attempt,
                StepStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                error_code=type(exc).__name__,
                error_message=str(exc)[:500],
                metadata={"download_duration_ms": duration_ms},
            )
            self._ingestion_repo.mark_job_failed(
                job.ingestion_job_id,
                failed_stage=IngestionStatus.DOWNLOADING,
                failure_reason=str(exc)[:2000],
            )
            logger.warning(
                "pdf acquisition failed paper_id=%s paper_version_id=%s "
                "ingestion_job_id=%s attempt=%d duration_ms=%d status=error error=%s",
                paper.paper_id,
                version.paper_version_id,
                job.ingestion_job_id,
                attempt,
                duration_ms,
                exc,
            )
            raise

        self._paper_repo.update_version_artifact(
            version.paper_version_id,
            checksum=checksum,
            storage_path=str(final_path),
            file_size_bytes=download_result.bytes_downloaded,
            downloaded_at=datetime.now(timezone.utc),
        )
        job = self._advance_job_to_downloaded(job)

        duration_ms = int((time.monotonic() - started_monotonic) * 1000)
        self._record_step(
            job.ingestion_job_id,
            attempt,
            StepStatus.COMPLETED,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            metadata={
                "bytes_downloaded": download_result.bytes_downloaded,
                "checksum": checksum,
                "download_duration_ms": duration_ms,
            },
        )
        logger.info(
            "pdf acquisition succeeded paper_id=%s paper_version_id=%s "
            "ingestion_job_id=%s attempt=%d bytes_downloaded=%d duration_ms=%d status=ok",
            paper.paper_id,
            version.paper_version_id,
            job.ingestion_job_id,
            attempt,
            download_result.bytes_downloaded,
            duration_ms,
        )
        return AcquisitionResult(job=job, artifact_reused=False)

    # --- Steps -------------------------------------------------------------

    def _next_attempt_number(self, ingestion_job_id: str) -> int:
        existing = [
            step
            for step in self._ingestion_repo.list_steps(ingestion_job_id)
            if step.stage == ProcessingStage.DOWNLOAD
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
                stage=ProcessingStage.DOWNLOAD,
                status=status,
                attempt=attempt,
                started_at=started_at,
                completed_at=completed_at,
                error_code=error_code,
                error_message=error_message,
                metadata=metadata or {},
            )
        )


def _validate_pdf_signature(path: Path) -> None:
    """Lightweight validation only (prompt #13): non-empty and starts with
    the PDF magic bytes. No deep parsing -- that's a later stage."""

    if path.stat().st_size == 0:
        raise InvalidPdfError("downloaded file is empty")
    with open(path, "rb") as fh:
        header = fh.read(len(_PDF_SIGNATURE))
    if header != _PDF_SIGNATURE:
        raise InvalidPdfError("downloaded file does not start with the expected PDF signature")
