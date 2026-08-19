"""Ingestion status domain models and the ingestion state machine.

``IngestionState`` (Prompt 1) is kept exactly as it was -- a lightweight
status snapshot -- rather than being modified to fit this stage's needs.
``IngestionJob`` *extends* it (adds identity + operational timestamps)
instead of changing it, so the Prompt 1 contract and its tests are
untouched while the PostgreSQL control plane gets the fuller entity it
needs to persist.

The state machine (``can_transition`` / ``validate_transition``) and the
resume-point calculation are pure functions with no database dependency,
per this stage's instruction to keep transition logic out of the ORM.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.core.exceptions import InvalidIngestionTransitionError
from app.domain.enums import IngestionStatus, ProcessingStage, StepStatus
from app.domain.ids import ensure_json_safe


class IngestionState(BaseModel):
    """Current processing status of a paper version in the ingestion pipeline."""

    paper_id: str
    paper_version_id: str
    status: IngestionStatus
    failed_stage: IngestionStatus | None = None
    failure_reason: str | None = None
    retry_count: int = 0

    @model_validator(mode="after")
    def _retry_count_non_negative(self) -> "IngestionState":
        if self.retry_count < 0:
            raise ValueError("retry_count must be >= 0")
        return self

    @model_validator(mode="after")
    def _failure_fields_consistent(self) -> "IngestionState":
        """A FAILED state must record which stage failed, so retries know
        where to resume from instead of repeating completed work."""

        if self.status == IngestionStatus.FAILED and self.failed_stage is None:
            raise ValueError("failed_stage is required when status is FAILED")
        return self


class IngestionJob(IngestionState):
    """A persisted ingestion job: an `IngestionState` snapshot plus the
    job's own identity and operational timestamps.

    Every validation rule inherited from `IngestionState` (non-negative
    `retry_count`, `failed_stage` required when FAILED) applies here too.
    """

    ingestion_job_id: str
    pipeline_version: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime | None = None


class IngestionStepState(BaseModel):
    """A single attempt of one processing stage within an ingestion job."""

    step_id: int | None = None
    ingestion_job_id: str
    stage: ProcessingStage
    status: StepStatus
    attempt: int = 1
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _attempt_positive(self) -> "IngestionStepState":
        if self.attempt < 1:
            raise ValueError("attempt must be >= 1")
        return self

    @model_validator(mode="after")
    def _metadata_json_safe(self) -> "IngestionStepState":
        ensure_json_safe(self.metadata)
        return self


# --- Ingestion state machine ------------------------------------------------
#
# READY and FAILED are terminal for this stage: this prompt explicitly does
# not implement automatic retry execution or an administrative recovery
# path, so nothing transitions *out of* FAILED yet (CLAUDE.md #18, prompt
# #10/#38). A later stage can extend `_LINEAR_TRANSITIONS`/`TERMINAL_STATUSES`
# when retry execution is actually implemented.

TERMINAL_STATUSES: frozenset[IngestionStatus] = frozenset(
    {IngestionStatus.READY, IngestionStatus.FAILED}
)

# The single linear happy-path: current status -> the one non-FAILED status
# it may advance to.
_LINEAR_TRANSITIONS: dict[IngestionStatus, IngestionStatus] = {
    IngestionStatus.DISCOVERED: IngestionStatus.DOWNLOADING,
    IngestionStatus.DOWNLOADING: IngestionStatus.DOWNLOADED,
    IngestionStatus.DOWNLOADED: IngestionStatus.PARSING,
    IngestionStatus.PARSING: IngestionStatus.PARSED,
    IngestionStatus.PARSED: IngestionStatus.CHUNKING,
    IngestionStatus.CHUNKING: IngestionStatus.CHUNKED,
    IngestionStatus.CHUNKED: IngestionStatus.VECTOR_INDEXING,
    IngestionStatus.VECTOR_INDEXING: IngestionStatus.VECTOR_INDEXED,
    IngestionStatus.VECTOR_INDEXED: IngestionStatus.GRAPH_INDEXING,
    IngestionStatus.GRAPH_INDEXING: IngestionStatus.GRAPH_INDEXED,
    IngestionStatus.GRAPH_INDEXED: IngestionStatus.READY,
}


def allowed_transitions(current: IngestionStatus) -> frozenset[IngestionStatus]:
    """Return the set of statuses `current` may legally move to.

    Every active (non-terminal) status may always move to FAILED, in
    addition to its one linear next status.
    """

    if current in TERMINAL_STATUSES:
        return frozenset()
    targets = {IngestionStatus.FAILED}
    next_status = _LINEAR_TRANSITIONS.get(current)
    if next_status is not None:
        targets.add(next_status)
    return frozenset(targets)


def can_transition(current: IngestionStatus, target: IngestionStatus) -> bool:
    """Check whether `current -> target` is a legal state transition."""

    return target in allowed_transitions(current)


def validate_transition(current: IngestionStatus, target: IngestionStatus) -> None:
    """Raise `InvalidIngestionTransitionError` if `current -> target` is illegal."""

    if not can_transition(current, target):
        raise InvalidIngestionTransitionError(
            f"cannot transition ingestion status from '{current.value}' to "
            f"'{target.value}'"
        )


# --- Resume-point calculation ------------------------------------------------
#
# Maps a job's current status to the next `ProcessingStage` that must run.
# Both the "-ING" (in-progress, presumably interrupted) and "-ED" (cleanly
# completed) statuses for a stage resolve to the *next* stage, since an
# in-progress stage found on resume must simply be re-attempted from
# scratch (idempotent processing stages are a later stage's job to ensure).
#
# GRAPH_EXTRACT is a valid `ProcessingStage` (for `ingestion_steps` rows)
# but deliberately does not appear here: at the job-status granularity,
# VECTOR_INDEXED's next stage is GRAPH_INDEX (matching this stage's given
# resume examples exactly) -- graph extraction and graph indexing are both
# folded under the single `GRAPH_INDEXING` job status. Finer-grained
# extract-vs-index attempts are distinguished in `ingestion_steps`, not in
# this coarse job-level mapping.
_NEXT_STAGE_BY_STATUS: dict[IngestionStatus, ProcessingStage | None] = {
    IngestionStatus.DISCOVERED: ProcessingStage.DOWNLOAD,
    IngestionStatus.DOWNLOADING: ProcessingStage.DOWNLOAD,
    IngestionStatus.DOWNLOADED: ProcessingStage.PARSE,
    IngestionStatus.PARSING: ProcessingStage.PARSE,
    IngestionStatus.PARSED: ProcessingStage.CHUNK,
    IngestionStatus.CHUNKING: ProcessingStage.CHUNK,
    IngestionStatus.CHUNKED: ProcessingStage.VECTOR_INDEX,
    IngestionStatus.VECTOR_INDEXING: ProcessingStage.VECTOR_INDEX,
    IngestionStatus.VECTOR_INDEXED: ProcessingStage.GRAPH_INDEX,
    IngestionStatus.GRAPH_INDEXING: ProcessingStage.GRAPH_INDEX,
    IngestionStatus.GRAPH_INDEXED: None,
    IngestionStatus.READY: None,
}


def get_resume_point(
    status: IngestionStatus, failed_stage: IngestionStatus | None = None
) -> ProcessingStage | None:
    """Determine the next processing stage a job should (re)run.

    For an active status this is simply the next stage after `status`. For
    FAILED, resume from wherever `failed_stage` left off instead of
    restarting the whole pipeline (CLAUDE.md #13). Returns ``None`` when
    there is nothing left to run (`GRAPH_INDEXED` awaiting the final
    `-> READY` transition, or an already-`READY` job).
    """

    if status == IngestionStatus.FAILED:
        if failed_stage is None:
            raise ValueError("failed_stage is required to resume a FAILED job")
        return _NEXT_STAGE_BY_STATUS.get(failed_stage)
    return _NEXT_STAGE_BY_STATUS.get(status)
