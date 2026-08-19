"""Integration tests for `POST /api/v1/papers/{paper_id}/ingest` and
`GET /api/v1/ingestion/{ingestion_job_id}`.

HTTP to the PDF source is mocked (`httpx.MockTransport`); PostgreSQL is
real (transaction-isolated via `db_session`), and `PAPER_STORAGE_PATH`
points at `tmp_path`. Requires a reachable database; skipped automatically
otherwise.
"""

from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.domain.papers import Paper, PaperVersion
from app.ingestion.download.client import PdfDownloadClient, get_pdf_download_client
from app.main import app
from app.storage.postgres.repositories.papers import PaperRepository
from app.storage.postgres.session import get_db_session

_FAKE_PDF = b"%PDF-1.4\n%fake pdf content\n%%EOF"


@pytest.fixture
def client(db_session, tmp_path, monkeypatch) -> Iterator[TestClient]:
    monkeypatch.setenv("PAPER_STORAGE_PATH", str(tmp_path))
    get_settings.cache_clear()

    def override_get_db_session():
        yield db_session

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_FAKE_PDF, headers={"content-type": "application/pdf"})

    fake_client = PdfDownloadClient(
        get_settings(), transport=httpx.MockTransport(handler), sleep=lambda _: None
    )

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_pdf_download_client] = lambda: fake_client
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db_session, None)
        app.dependency_overrides.pop(get_pdf_download_client, None)
        get_settings.cache_clear()


def _discover_paper(db_session, source_id: str = "2401.66601") -> Paper:
    papers = PaperRepository(db_session)
    paper = papers.upsert_paper(
        Paper.create(
            source="arxiv",
            source_id=source_id,
            title="A Paper",
            pdf_url=f"https://arxiv.org/pdf/{source_id}v1",
        )
    )
    papers.get_or_create_paper_version(PaperVersion.create(paper_id=paper.paper_id, version="v1"))
    return paper


class TestIngestEndpoint:
    def test_ingest_downloads_and_returns_downloaded_status(
        self, client: TestClient, db_session
    ) -> None:
        paper = _discover_paper(db_session)

        response = client.post(f"/api/v1/papers/{paper.paper_id}/ingest")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "downloaded"
        assert body["artifact_available"] is True
        assert body["artifact_reused"] is False
        assert body["paper_id"] == paper.paper_id
        assert body["ingestion_job_id"]

    def test_second_ingest_reuses_the_artifact(self, client: TestClient, db_session) -> None:
        paper = _discover_paper(db_session, source_id="2401.66602")

        client.post(f"/api/v1/papers/{paper.paper_id}/ingest")
        response = client.post(f"/api/v1/papers/{paper.paper_id}/ingest")

        assert response.status_code == 200
        assert response.json()["artifact_reused"] is True

    def test_ingest_for_unknown_paper_returns_404(self, client: TestClient) -> None:
        response = client.post("/api/v1/papers/paper:arxiv:9999.99999/ingest")
        assert response.status_code == 404

    def test_ingest_with_explicit_paper_version_id(self, client: TestClient, db_session) -> None:
        paper = _discover_paper(db_session, source_id="2401.66603")
        version_id = f"paper-version:arxiv:{paper.source_id}:v1"

        response = client.post(
            f"/api/v1/papers/{paper.paper_id}/ingest", json={"paper_version_id": version_id}
        )

        assert response.status_code == 200
        assert response.json()["paper_version_id"] == version_id


class TestIngestionStatusEndpoint:
    def test_status_reflects_the_ingest_result(self, client: TestClient, db_session) -> None:
        paper = _discover_paper(db_session, source_id="2401.66604")
        ingest_response = client.post(f"/api/v1/papers/{paper.paper_id}/ingest")
        job_id = ingest_response.json()["ingestion_job_id"]

        response = client.get(f"/api/v1/ingestion/{job_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["ingestion_job_id"] == job_id
        assert body["status"] == "downloaded"
        assert body["retry_count"] == 0
        assert body["started_at"] is not None
        # `completed_at` means the whole ingestion job finished, which only
        # happens at READY (a later, not-yet-implemented stage) -- reaching
        # DOWNLOADED must not set it.
        assert body["completed_at"] is None

    def test_status_for_unknown_job_returns_404(self, client: TestClient) -> None:
        response = client.get("/api/v1/ingestion/ingestion-job:doesnotexist")
        assert response.status_code == 404

    def test_health_still_works_alongside_ingestion_routes(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
