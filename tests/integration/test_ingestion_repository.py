"""Integration tests for `IngestionRepository` against real PostgreSQL.

Requires a reachable database (see `tests/integration/conftest.py`);
skipped automatically otherwise.
"""

import pytest
from sqlalchemy.exc import IntegrityError

from app.core.exceptions import InvalidIngestionTransitionError
from app.domain.enums import IngestionStatus, ProcessingStage, StepStatus
from app.domain.ids import build_ingestion_job_id
from app.domain.ingestion import IngestionJob, IngestionStepState
from app.domain.papers import Paper, PaperVersion
from app.storage.postgres.mappings import domain_ingestion_job_to_record
from app.storage.postgres.repositories.ingestion import IngestionRepository
from app.storage.postgres.repositories.papers import PaperRepository

_FULL_HAPPY_PATH = [
    IngestionStatus.DOWNLOADING,
    IngestionStatus.DOWNLOADED,
    IngestionStatus.PARSING,
    IngestionStatus.PARSED,
    IngestionStatus.CHUNKING,
    IngestionStatus.CHUNKED,
    IngestionStatus.VECTOR_INDEXING,
    IngestionStatus.VECTOR_INDEXED,
    IngestionStatus.GRAPH_INDEXING,
    IngestionStatus.GRAPH_INDEXED,
    IngestionStatus.READY,
]


@pytest.fixture
def paper_version(db_session):
    papers = PaperRepository(db_session)
    paper = papers.upsert_paper(
        Paper.create(source="arxiv", source_id="2401.99999", title="A Paper")
    )
    return papers.get_or_create_paper_version(
        PaperVersion.create(paper_id=paper.paper_id, version="1")
    )


class TestCreateIngestionJob:
    def test_create_job_persists_it_in_discovered_status(self, db_session, paper_version) -> None:
        repo = IngestionRepository(db_session)
        job = repo.create_ingestion_job(
            paper_id=paper_version.paper_id, paper_version_id=paper_version.paper_version_id
        )

        assert job.status == IngestionStatus.DISCOVERED
        assert repo.get_job(job.ingestion_job_id).ingestion_job_id == job.ingestion_job_id

    def test_second_call_reuses_the_active_job(self, db_session, paper_version) -> None:
        repo = IngestionRepository(db_session)
        first = repo.create_ingestion_job(
            paper_id=paper_version.paper_id, paper_version_id=paper_version.paper_version_id
        )
        second = repo.create_ingestion_job(
            paper_id=paper_version.paper_id, paper_version_id=paper_version.paper_version_id
        )

        assert first.ingestion_job_id == second.ingestion_job_id

    def test_database_constraint_blocks_two_active_jobs_for_same_version(
        self, db_session, paper_version
    ) -> None:
        # Bypass the repository's own check-then-insert idempotency logic
        # (as a real race between two concurrent requests would) to prove
        # the partial unique index -- not just application logic -- is
        # what ultimately guarantees at most one active job per paper
        # version (CLAUDE.md #28).
        job_a = IngestionJob(
            ingestion_job_id=build_ingestion_job_id(),
            paper_id=paper_version.paper_id,
            paper_version_id=paper_version.paper_version_id,
            status=IngestionStatus.DISCOVERED,
        )
        job_b = IngestionJob(
            ingestion_job_id=build_ingestion_job_id(),
            paper_id=paper_version.paper_id,
            paper_version_id=paper_version.paper_version_id,
            status=IngestionStatus.DISCOVERED,
        )
        db_session.add(domain_ingestion_job_to_record(job_a))
        db_session.flush()
        db_session.add(domain_ingestion_job_to_record(job_b))
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_new_job_created_after_previous_one_reached_ready(self, db_session, paper_version) -> None:
        repo = IngestionRepository(db_session)
        first = repo.create_ingestion_job(
            paper_id=paper_version.paper_id, paper_version_id=paper_version.paper_version_id
        )
        for target in _FULL_HAPPY_PATH:
            repo.transition_job_status(first.ingestion_job_id, target)

        assert repo.get_active_job_for_version(paper_version.paper_version_id) is None
        assert repo.is_version_ready(paper_version.paper_version_id) is True

        second = repo.create_ingestion_job(
            paper_id=paper_version.paper_id, paper_version_id=paper_version.paper_version_id
        )
        assert second.ingestion_job_id != first.ingestion_job_id


class TestTransitionJobStatus:
    def test_valid_transition_updates_status_and_sets_started_at(self, db_session, paper_version) -> None:
        repo = IngestionRepository(db_session)
        job = repo.create_ingestion_job(
            paper_id=paper_version.paper_id, paper_version_id=paper_version.paper_version_id
        )

        updated = repo.transition_job_status(job.ingestion_job_id, IngestionStatus.DOWNLOADING)

        assert updated.status == IngestionStatus.DOWNLOADING
        assert updated.started_at is not None

    def test_reaching_ready_sets_completed_at(self, db_session, paper_version) -> None:
        repo = IngestionRepository(db_session)
        job = repo.create_ingestion_job(
            paper_id=paper_version.paper_id, paper_version_id=paper_version.paper_version_id
        )

        updated = job
        for target in _FULL_HAPPY_PATH:
            updated = repo.transition_job_status(job.ingestion_job_id, target)

        assert updated.status == IngestionStatus.READY
        assert updated.completed_at is not None

    def test_invalid_transition_is_rejected_and_status_unchanged(self, db_session, paper_version) -> None:
        repo = IngestionRepository(db_session)
        job = repo.create_ingestion_job(
            paper_id=paper_version.paper_id, paper_version_id=paper_version.paper_version_id
        )

        with pytest.raises(InvalidIngestionTransitionError):
            repo.transition_job_status(job.ingestion_job_id, IngestionStatus.READY)

        assert repo.get_job(job.ingestion_job_id).status == IngestionStatus.DISCOVERED


class TestMarkJobFailed:
    def test_mark_failed_records_stage_reason_and_increments_retry_count(
        self, db_session, paper_version
    ) -> None:
        repo = IngestionRepository(db_session)
        job = repo.create_ingestion_job(
            paper_id=paper_version.paper_id, paper_version_id=paper_version.paper_version_id
        )
        repo.transition_job_status(job.ingestion_job_id, IngestionStatus.DOWNLOADING)
        repo.transition_job_status(job.ingestion_job_id, IngestionStatus.DOWNLOADED)
        repo.transition_job_status(job.ingestion_job_id, IngestionStatus.PARSING)

        failed = repo.mark_job_failed(
            job.ingestion_job_id,
            failed_stage=IngestionStatus.PARSING,
            failure_reason="malformed PDF",
        )

        assert failed.status == IngestionStatus.FAILED
        assert failed.failed_stage == IngestionStatus.PARSING
        assert failed.failure_reason == "malformed PDF"
        assert failed.retry_count == 1

    def test_failed_status_cannot_transition_further(self, db_session, paper_version) -> None:
        repo = IngestionRepository(db_session)
        job = repo.create_ingestion_job(
            paper_id=paper_version.paper_id, paper_version_id=paper_version.paper_version_id
        )
        repo.mark_job_failed(
            job.ingestion_job_id,
            failed_stage=IngestionStatus.DISCOVERED,
            failure_reason="arXiv unreachable",
        )

        with pytest.raises(InvalidIngestionTransitionError):
            repo.transition_job_status(job.ingestion_job_id, IngestionStatus.DOWNLOADING)


class TestResumePoint:
    def test_resume_point_reflects_persisted_status(self, db_session, paper_version) -> None:
        repo = IngestionRepository(db_session)
        job = repo.create_ingestion_job(
            paper_id=paper_version.paper_id, paper_version_id=paper_version.paper_version_id
        )
        for target in [
            IngestionStatus.DOWNLOADING,
            IngestionStatus.DOWNLOADED,
            IngestionStatus.PARSING,
            IngestionStatus.PARSED,
        ]:
            repo.transition_job_status(job.ingestion_job_id, target)

        assert repo.get_resume_point(job.ingestion_job_id) == ProcessingStage.CHUNK

    def test_resume_point_after_failure_uses_failed_stage_not_failed_status(
        self, db_session, paper_version
    ) -> None:
        repo = IngestionRepository(db_session)
        job = repo.create_ingestion_job(
            paper_id=paper_version.paper_id, paper_version_id=paper_version.paper_version_id
        )
        repo.transition_job_status(job.ingestion_job_id, IngestionStatus.DOWNLOADING)
        repo.mark_job_failed(
            job.ingestion_job_id,
            failed_stage=IngestionStatus.DOWNLOADING,
            failure_reason="network timeout",
        )

        assert repo.get_resume_point(job.ingestion_job_id) == ProcessingStage.DOWNLOAD


class TestIngestionSteps:
    def test_record_and_list_steps(self, db_session, paper_version) -> None:
        repo = IngestionRepository(db_session)
        job = repo.create_ingestion_job(
            paper_id=paper_version.paper_id, paper_version_id=paper_version.paper_version_id
        )

        repo.record_step(
            IngestionStepState(
                ingestion_job_id=job.ingestion_job_id,
                stage=ProcessingStage.DOWNLOAD,
                status=StepStatus.COMPLETED,
                attempt=1,
            )
        )
        steps = repo.list_steps(job.ingestion_job_id)

        assert len(steps) == 1
        assert steps[0].stage == ProcessingStage.DOWNLOAD
        assert steps[0].status == StepStatus.COMPLETED

    def test_recording_the_same_stage_and_attempt_again_updates_in_place(
        self, db_session, paper_version
    ) -> None:
        repo = IngestionRepository(db_session)
        job = repo.create_ingestion_job(
            paper_id=paper_version.paper_id, paper_version_id=paper_version.paper_version_id
        )

        repo.record_step(
            IngestionStepState(
                ingestion_job_id=job.ingestion_job_id,
                stage=ProcessingStage.DOWNLOAD,
                status=StepStatus.IN_PROGRESS,
                attempt=1,
            )
        )
        repo.record_step(
            IngestionStepState(
                ingestion_job_id=job.ingestion_job_id,
                stage=ProcessingStage.DOWNLOAD,
                status=StepStatus.COMPLETED,
                attempt=1,
            )
        )

        steps = repo.list_steps(job.ingestion_job_id)
        assert len(steps) == 1
        assert steps[0].status == StepStatus.COMPLETED
