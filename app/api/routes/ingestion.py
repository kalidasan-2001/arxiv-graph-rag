"""Ingestion job status route.

Job-oriented (`/ingestion/{ingestion_job_id}`), not paper-oriented, because
`ingestion_job_id` is already the stable identifier every write in this
package (Prompt 2, Prompt 4) returns and operates on -- a paper-oriented
`/papers/{paper_id}/status` would still have to pick "which job" if a
paper has ingestion history, so the job id is the cleaner existing
convention to key off of (prompt #6).
"""

from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.storage.postgres.repositories.ingestion import IngestionRepository
from app.storage.postgres.session import get_db_session

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


class IngestionStatusResponse(BaseModel):
    """Operational state of one ingestion job. Never a raw stack trace --
    `failure_reason` is the concise, persisted diagnostic text only."""

    ingestion_job_id: str
    paper_id: str
    paper_version_id: str
    status: str
    failed_stage: str | None
    failure_reason: str | None
    retry_count: int
    started_at: datetime | None
    completed_at: datetime | None


@router.get("/{ingestion_job_id}", response_model=IngestionStatusResponse)
def get_ingestion_status(
    ingestion_job_id: str, session: Session = Depends(get_db_session)
) -> IngestionStatusResponse:
    job = IngestionRepository(session).get_job(ingestion_job_id)

    return IngestionStatusResponse(
        ingestion_job_id=job.ingestion_job_id,
        paper_id=job.paper_id,
        paper_version_id=job.paper_version_id,
        status=job.status.value,
        failed_stage=job.failed_stage.value if job.failed_stage else None,
        failure_reason=job.failure_reason,
        retry_count=job.retry_count,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )
