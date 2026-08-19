"""Integration tests for `PdfAcquisitionService` against real PostgreSQL.

HTTP is always mocked (`httpx.MockTransport`) -- these are deterministic
and never touch live arXiv. Requires a reachable database (see
`tests/integration/conftest.py`); skipped automatically otherwise.

Covers the full target flow (prompt #55):
discovered paper -> explicit ingest -> job -> DOWNLOADING -> filesystem
artifact -> checksum persisted -> DOWNLOADED -- plus idempotency,
reconciliation after a simulated partial DB failure, and corrupt-artifact
recovery.
"""

import httpx
import pytest
from sqlalchemy import select

from app.core.config import Settings
from app.core.exceptions import PaperVersionNotFoundError
from app.domain.enums import IngestionStatus
from app.domain.papers import Paper, PaperVersion
from app.ingestion.checksums import sha256_file
from app.ingestion.download.client import PdfDownloadClient
from app.ingestion.download.service import PdfAcquisitionService
from app.ingestion.download.storage import PaperStorage
from app.storage.postgres.models import IngestionJobRecord
from app.storage.postgres.repositories.ingestion import IngestionRepository
from app.storage.postgres.repositories.papers import PaperRepository

_FAKE_PDF = b"%PDF-1.4\n%fake pdf content for integration testing\n%%EOF"


def _settings(storage_root, **overrides) -> Settings:
    defaults = dict(
        PAPER_STORAGE_PATH=str(storage_root),
        PDF_DOWNLOAD_TIMEOUT_SECONDS=5,
        PDF_DOWNLOAD_MAX_RETRIES=1,
        MAX_PAPER_SIZE_MB=10,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _counting_client(settings: Settings, body: bytes = _FAKE_PDF, status: int = 200) -> tuple[PdfDownloadClient, dict]:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(status, content=body, headers={"content-type": "application/pdf"})

    client = PdfDownloadClient(settings, transport=httpx.MockTransport(handler), sleep=lambda _: None)
    return client, calls


def _discover_paper(db_session, *, source_id: str = "2401.55501") -> tuple[Paper, PaperVersion]:
    papers = PaperRepository(db_session)
    paper = papers.upsert_paper(
        Paper.create(
            source="arxiv",
            source_id=source_id,
            title="A Paper",
            pdf_url=f"https://arxiv.org/pdf/{source_id}v1",
        )
    )
    # "v1", not "1" -- matches the normalized form arXiv discovery actually
    # produces (`normalize_arxiv_id`), which is also what the storage path
    # directory name is built from.
    version = papers.get_or_create_paper_version(
        PaperVersion.create(paper_id=paper.paper_id, version="v1")
    )
    return paper, version


def _service(db_session, client: PdfDownloadClient, settings: Settings) -> PdfAcquisitionService:
    storage = PaperStorage(settings)
    return PdfAcquisitionService(
        settings, client, storage, PaperRepository(db_session), IngestionRepository(db_session)
    )


class TestFullAcquisitionFlow:
    def test_discovered_paper_is_downloaded_validated_and_marked_downloaded(
        self, db_session, tmp_path
    ) -> None:
        paper, version = _discover_paper(db_session)
        settings = _settings(tmp_path)
        client, calls = _counting_client(settings)

        result = _service(db_session, client, settings).ingest(paper.paper_id)

        assert result.job.status == IngestionStatus.DOWNLOADED
        assert result.artifact_reused is False
        assert calls["count"] == 1

        stored_version = PaperRepository(db_session).get_paper_version(version.paper_version_id)
        assert stored_version.checksum is not None
        assert stored_version.storage_path is not None
        assert stored_version.file_size_bytes == len(_FAKE_PDF)
        assert stored_version.downloaded_at is not None

        artifact_path = tmp_path / "arxiv" / paper.source_id / "v1" / "paper.pdf"
        assert artifact_path.is_file()
        assert artifact_path.read_bytes() == _FAKE_PDF
        assert sha256_file(artifact_path) == stored_version.checksum

    def test_no_version_specified_uses_the_latest_discovered_version(
        self, db_session, tmp_path
    ) -> None:
        papers = PaperRepository(db_session)
        paper = papers.upsert_paper(
            Paper.create(
                source="arxiv",
                source_id="2401.55502",
                title="A Paper",
                pdf_url="https://arxiv.org/pdf/2401.55502v3",
            )
        )
        papers.get_or_create_paper_version(PaperVersion.create(paper_id=paper.paper_id, version="v1"))
        papers.get_or_create_paper_version(PaperVersion.create(paper_id=paper.paper_id, version="v3"))
        papers.get_or_create_paper_version(PaperVersion.create(paper_id=paper.paper_id, version="v2"))

        settings = _settings(tmp_path)
        client, _ = _counting_client(settings)

        result = _service(db_session, client, settings).ingest(paper.paper_id)

        assert result.job.paper_version_id == "paper-version:arxiv:2401.55502:v3"

    def test_missing_version_for_undiscovered_paper_raises(self, db_session, tmp_path) -> None:
        papers = PaperRepository(db_session)
        paper = papers.upsert_paper(
            Paper.create(source="arxiv", source_id="2401.55503", title="No Versions")
        )
        settings = _settings(tmp_path)
        client, _ = _counting_client(settings)

        with pytest.raises(PaperVersionNotFoundError):
            _service(db_session, client, settings).ingest(paper.paper_id)


class TestIdempotency:
    def test_second_ingest_does_not_perform_a_second_http_download(
        self, db_session, tmp_path
    ) -> None:
        paper, _ = _discover_paper(db_session, source_id="2401.55510")
        settings = _settings(tmp_path)
        client, calls = _counting_client(settings)
        service = _service(db_session, client, settings)

        first = service.ingest(paper.paper_id)
        second = service.ingest(paper.paper_id)

        assert calls["count"] == 1  # the critical assertion: no second HTTP call
        assert first.job.status == IngestionStatus.DOWNLOADED
        assert second.artifact_reused is True
        assert second.job.status == IngestionStatus.DOWNLOADED
        assert second.job.ingestion_job_id == first.job.ingestion_job_id

    def test_repeated_ingest_does_not_duplicate_ingestion_jobs(self, db_session, tmp_path) -> None:
        paper, version = _discover_paper(db_session, source_id="2401.55511")
        settings = _settings(tmp_path)
        client, _ = _counting_client(settings)
        service = _service(db_session, client, settings)

        service.ingest(paper.paper_id)
        service.ingest(paper.paper_id)
        service.ingest(paper.paper_id)

        rows = db_session.execute(
            select(IngestionJobRecord).where(
                IngestionJobRecord.paper_version_id == version.paper_version_id
            )
        ).scalars().all()
        assert len(rows) == 1


class TestPartialFailureReconciliation:
    def test_artifact_finalized_but_final_db_write_failed_is_reconciled_without_redownload(
        self, db_session, tmp_path
    ) -> None:
        """Simulates prompt #27/#46: the file was written and validated,
        but the write that would have recorded checksum/storage_path on
        the paper_version never happened (process crash, DB blip, ...).
        The next call must detect the artifact via the filesystem, not
        require a second HTTP download, and reconcile PostgreSQL."""

        paper, version = _discover_paper(db_session, source_id="2401.55520")
        settings = _settings(tmp_path)

        # Manually place a valid artifact at the deterministic path,
        # simulating "download succeeded, DB write did not" -- without
        # ever calling the service (so no checksum/storage_path exists in
        # the DB yet, and no ingestion job exists yet either).
        storage = PaperStorage(settings)
        temp_path = storage.get_temp_path(source="arxiv", source_id=paper.source_id, version="v1")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(_FAKE_PDF)
        storage.finalize(temp_path, source="arxiv", source_id=paper.source_id, version="v1")

        client, calls = _counting_client(settings)
        result = _service(db_session, client, settings).ingest(paper.paper_id)

        assert calls["count"] == 0  # never re-downloaded -- reconciled from disk
        assert result.artifact_reused is True
        assert result.job.status == IngestionStatus.DOWNLOADED

        stored_version = PaperRepository(db_session).get_paper_version(version.paper_version_id)
        assert stored_version.checksum == sha256_file(
            storage.get_path(source="arxiv", source_id=paper.source_id, version="v1")
        )


class TestCorruptArtifactRecovery:
    def test_corrupted_artifact_is_detected_and_a_fresh_valid_copy_is_downloaded(
        self, db_session, tmp_path
    ) -> None:
        """Simulates prompt #47: PostgreSQL says DOWNLOADED (checksum
        recorded), but the file on disk has since been corrupted. The
        service must NOT claim success based on the corrupt data -- it
        must detect the mismatch, discard it, and fetch a fresh, valid
        artifact before reporting DOWNLOADED."""

        paper, version = _discover_paper(db_session, source_id="2401.55530")
        settings = _settings(tmp_path)
        client, calls = _counting_client(settings)
        service = _service(db_session, client, settings)

        first = service.ingest(paper.paper_id)
        assert first.job.status == IngestionStatus.DOWNLOADED
        original_checksum = PaperRepository(db_session).get_paper_version(
            version.paper_version_id
        ).checksum

        # Corrupt the artifact in place.
        storage = PaperStorage(settings)
        artifact_path = storage.get_path(source="arxiv", source_id=paper.source_id, version="v1")
        artifact_path.write_bytes(b"%PDF-1.4 CORRUPTED DATA, NOT THE ORIGINAL")

        second = service.ingest(paper.paper_id)

        assert calls["count"] == 2  # corruption forced a real second download
        assert second.job.status == IngestionStatus.DOWNLOADED
        assert second.job.ingestion_job_id != first.job.ingestion_job_id  # stale job replaced

        final_version = PaperRepository(db_session).get_paper_version(version.paper_version_id)
        assert final_version.checksum == original_checksum  # back to the real content's checksum
        assert artifact_path.read_bytes() == _FAKE_PDF  # corrupt bytes were overwritten
