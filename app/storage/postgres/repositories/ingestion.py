"""Ingestion job and step persistence: the operational control plane.

Active-job policy (prompt #14): an "active" job is one whose status is not
in `TERMINAL_STATUSES` (READY, FAILED). `create_ingestion_job` reuses an
existing active job for the same paper version instead of creating a
duplicate. That reuse check alone is application logic and racy under
concurrency (CLAUDE.md #28: two requests could both see "no active job"
and both insert), so it's backed by a partial unique index --
`uq_ingestion_jobs_one_active_per_version` on `ingestion_jobs` --
enforcing "at most one active job per paper version" at the database
level. More than one *terminal* (READY/FAILED) job per paper version over
time is legitimate (e.g. a failed run followed by a successful one), so
the index only applies to non-terminal rows. `create_ingestion_job`
catches the resulting `IntegrityError` and returns whichever job actually
won the race.

Retry-count policy: `retry_count` increments each time `mark_job_failed` is
called, i.e. it counts recorded failed attempts. Automatic retry execution
is explicitly out of scope for this stage (prompt #18); a future stage will
compare `retry_count` against a configured `MAX_RETRY` limit before
deciding whether to start a new attempt.
"""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.exceptions import IngestionJobNotFoundError
from app.domain.enums import IngestionStatus, ProcessingStage
from app.domain.ids import build_ingestion_job_id
from app.domain.ingestion import (
    TERMINAL_STATUSES,
    IngestionJob,
    IngestionStepState,
    get_resume_point,
    validate_transition,
)
from app.storage.postgres.mappings import (
    domain_ingestion_job_to_record,
    domain_step_to_record,
    ingestion_job_record_to_domain,
    step_record_to_domain,
    update_step_record_from_domain,
)
from app.storage.postgres.models import IngestionJobRecord, IngestionStepRecord

# Diagnostic text stored on `failure_reason` is capped so a runaway error
# message (or an accidentally-embedded stack trace) can't bloat the table;
# deeper detail belongs in logs, not persisted user-facing state (prompt #17).
_MAX_FAILURE_REASON_LENGTH = 2000

_TERMINAL_STATUS_VALUES = [status.value for status in TERMINAL_STATUSES]


class IngestionRepository:
    """Persistence + control-plane operations for ingestion jobs and steps."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # --- Jobs -------------------------------------------------------

    def get_job(self, ingestion_job_id: str) -> IngestionJob:
        record = self._session.get(IngestionJobRecord, ingestion_job_id)
        if record is None:
            raise IngestionJobNotFoundError(f"ingestion job {ingestion_job_id} not found")
        return ingestion_job_record_to_domain(record)

    def get_active_job_for_version(self, paper_version_id: str) -> IngestionJob | None:
        """Return the non-terminal job for this paper version, if any."""

        stmt = select(IngestionJobRecord).where(
            IngestionJobRecord.paper_version_id == paper_version_id,
            IngestionJobRecord.status.notin_(_TERMINAL_STATUS_VALUES),
        )
        record = self._session.execute(stmt).scalars().first()
        return ingestion_job_record_to_domain(record) if record else None

    def is_version_ready(self, paper_version_id: str) -> bool:
        """Check whether any job for this paper version has already reached READY.

        Lets callers detect "this paper version is already fully processed"
        (prompt #15) without creating a redundant ingestion job.
        """

        stmt = select(IngestionJobRecord.ingestion_job_id).where(
            IngestionJobRecord.paper_version_id == paper_version_id,
            IngestionJobRecord.status == IngestionStatus.READY.value,
        )
        return self._session.execute(stmt).first() is not None

    def create_ingestion_job(
        self,
        *,
        paper_id: str,
        paper_version_id: str,
        pipeline_version: str | None = None,
    ) -> IngestionJob:
        """Create a new ingestion job, or reuse an existing active one.

        Does not check `is_version_ready` itself -- callers decide whether
        reprocessing an already-READY version is intentional (prompt #15
        explicitly defers that policy).
        """

        existing = self.get_active_job_for_version(paper_version_id)
        if existing is not None:
            return existing

        job = IngestionJob(
            ingestion_job_id=build_ingestion_job_id(),
            paper_id=paper_id,
            paper_version_id=paper_version_id,
            status=IngestionStatus.DISCOVERED,
            pipeline_version=pipeline_version,
        )
        record = domain_ingestion_job_to_record(job)
        try:
            # A SAVEPOINT so a lost race (caught below) only discards this
            # insert attempt, not the caller's whole unit of work.
            with self._session.begin_nested():
                self._session.add(record)
                self._session.flush()
        except IntegrityError:
            # Lost a race to a concurrent create_ingestion_job call for the
            # same paper version -- `uq_ingestion_jobs_one_active_per_version`
            # is the actual source of truth here; return whichever job won.
            existing = self.get_active_job_for_version(paper_version_id)
            if existing is not None:
                return existing
            raise
        return ingestion_job_record_to_domain(record)

    def transition_job_status(
        self, ingestion_job_id: str, target_status: IngestionStatus
    ) -> IngestionJob:
        """Validate and apply a status transition, updating timestamps.

        `started_at` is set the first time a job leaves DISCOVERED (whether
        it advances normally or fails immediately). `completed_at` is set
        when the job reaches READY.
        """

        record = self._session.get(IngestionJobRecord, ingestion_job_id)
        if record is None:
            raise IngestionJobNotFoundError(f"ingestion job {ingestion_job_id} not found")

        current_status = IngestionStatus(record.status)
        validate_transition(current_status, target_status)

        now = datetime.now(timezone.utc)
        if record.started_at is None:
            record.started_at = now
        record.status = target_status.value
        if target_status == IngestionStatus.READY:
            record.completed_at = now
        self._session.flush()
        return ingestion_job_record_to_domain(record)

    def mark_job_failed(
        self,
        ingestion_job_id: str,
        *,
        failed_stage: IngestionStatus,
        failure_reason: str,
    ) -> IngestionJob:
        """Transition a job to FAILED and record concise diagnostic metadata."""

        record = self._session.get(IngestionJobRecord, ingestion_job_id)
        if record is None:
            raise IngestionJobNotFoundError(f"ingestion job {ingestion_job_id} not found")

        current_status = IngestionStatus(record.status)
        validate_transition(current_status, IngestionStatus.FAILED)

        now = datetime.now(timezone.utc)
        if record.started_at is None:
            record.started_at = now
        record.status = IngestionStatus.FAILED.value
        record.failed_stage = failed_stage.value
        record.failure_reason = failure_reason[:_MAX_FAILURE_REASON_LENGTH]
        record.retry_count += 1
        self._session.flush()
        return ingestion_job_record_to_domain(record)

    def get_resume_point(self, ingestion_job_id: str) -> ProcessingStage | None:
        """Determine the next processing stage this job should (re)run."""

        job = self.get_job(ingestion_job_id)
        return get_resume_point(job.status, job.failed_stage)

    # --- Steps -------------------------------------------------------

    def record_step(self, step: IngestionStepState) -> IngestionStepState:
        """Insert or update a step attempt row (unique per job + stage + attempt)."""

        stmt = select(IngestionStepRecord).where(
            IngestionStepRecord.ingestion_job_id == step.ingestion_job_id,
            IngestionStepRecord.stage == step.stage.value,
            IngestionStepRecord.attempt == step.attempt,
        )
        existing = self._session.execute(stmt).scalar_one_or_none()
        if existing is None:
            record = domain_step_to_record(step)
            self._session.add(record)
            self._session.flush()
            return step_record_to_domain(record)

        update_step_record_from_domain(existing, step)
        self._session.flush()
        return step_record_to_domain(existing)

    def list_steps(self, ingestion_job_id: str) -> list[IngestionStepState]:
        stmt = (
            select(IngestionStepRecord)
            .where(IngestionStepRecord.ingestion_job_id == ingestion_job_id)
            .order_by(IngestionStepRecord.step_id)
        )
        records = self._session.execute(stmt).scalars().all()
        return [step_record_to_domain(record) for record in records]
