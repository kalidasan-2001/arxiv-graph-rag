"""Integration tests for `POST /api/v1/papers/{paper_id}/parse` and
`GET /api/v1/papers/{paper_id}/document`.

No live network -- the "downloaded PDF" precondition is set up directly.
PostgreSQL is real (transaction-isolated via `db_session`); requires a
reachable database, skipped automatically otherwise.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.domain.papers import Paper, PaperVersion
from app.ingestion.checksums import sha256_file
from app.ingestion.download.storage import PaperStorage
from app.main import app
from app.storage.postgres.repositories.papers import PaperRepository
from app.storage.postgres.session import get_db_session
from tests.parsing.pdf_fixtures import make_scientific_paper_pdf_bytes


@pytest.fixture
def client(db_session, tmp_path, monkeypatch) -> Iterator[TestClient]:
    monkeypatch.setenv("PAPER_STORAGE_PATH", str(tmp_path))
    get_settings.cache_clear()

    def override_get_db_session():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db_session
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db_session, None)
        get_settings.cache_clear()


def _prepare_downloaded_paper(db_session, tmp_path, source_id: str = "2401.88801") -> Paper:
    papers = PaperRepository(db_session)
    paper = papers.upsert_paper(Paper.create(source="arxiv", source_id=source_id, title="A Paper"))
    version = papers.get_or_create_paper_version(
        PaperVersion.create(paper_id=paper.paper_id, version="v1")
    )

    from app.core.config import Settings

    settings = Settings(PAPER_STORAGE_PATH=str(tmp_path))
    pdf_storage = PaperStorage(settings)
    temp_path = pdf_storage.get_temp_path(source="arxiv", source_id=source_id, version="v1")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(make_scientific_paper_pdf_bytes())
    final_path = pdf_storage.finalize(temp_path, source="arxiv", source_id=source_id, version="v1")

    papers.update_version_artifact(
        version.paper_version_id,
        checksum=sha256_file(final_path),
        storage_path=str(final_path),
        file_size_bytes=final_path.stat().st_size,
        downloaded_at=version.created_at,
    )
    return paper


class TestParseEndpoint:
    def test_parse_returns_parsed_status_and_counts(self, client: TestClient, db_session, tmp_path) -> None:
        paper = _prepare_downloaded_paper(db_session, tmp_path)

        response = client.post(f"/api/v1/papers/{paper.paper_id}/parse")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "parsed"
        assert body["page_count"] == 3
        assert body["section_count"] > 0
        assert body["parse_reused"] is False

    def test_second_parse_reports_reused(self, client: TestClient, db_session, tmp_path) -> None:
        paper = _prepare_downloaded_paper(db_session, tmp_path, source_id="2401.88802")

        client.post(f"/api/v1/papers/{paper.paper_id}/parse")
        response = client.post(f"/api/v1/papers/{paper.paper_id}/parse")

        assert response.json()["parse_reused"] is True

    def test_parse_without_download_returns_502(self, client: TestClient, db_session) -> None:
        papers = PaperRepository(db_session)
        paper = papers.upsert_paper(
            Paper.create(source="arxiv", source_id="2401.88803", title="Not Downloaded")
        )
        papers.get_or_create_paper_version(PaperVersion.create(paper_id=paper.paper_id, version="v1"))

        response = client.post(f"/api/v1/papers/{paper.paper_id}/parse")

        assert response.status_code == 502  # InvalidPdfError's existing mapping

    def test_parse_for_unknown_paper_returns_404(self, client: TestClient) -> None:
        response = client.post("/api/v1/papers/paper:arxiv:9999.99999/parse")
        assert response.status_code == 404


class TestDocumentEndpoint:
    def test_document_returns_sections_without_text_by_default(
        self, client: TestClient, db_session, tmp_path
    ) -> None:
        paper = _prepare_downloaded_paper(db_session, tmp_path, source_id="2401.88810")
        client.post(f"/api/v1/papers/{paper.paper_id}/parse")

        response = client.get(f"/api/v1/papers/{paper.paper_id}/document")

        assert response.status_code == 200
        body = response.json()
        assert body["page_count"] == 3
        assert body["parser_name"] == "pymupdf"
        assert len(body["sections"]) > 0
        assert all(section["text"] is None for section in body["sections"])

    def test_document_includes_text_when_requested(
        self, client: TestClient, db_session, tmp_path
    ) -> None:
        paper = _prepare_downloaded_paper(db_session, tmp_path, source_id="2401.88811")
        client.post(f"/api/v1/papers/{paper.paper_id}/parse")

        response = client.get(
            f"/api/v1/papers/{paper.paper_id}/document", params={"include_text": "true"}
        )

        body = response.json()
        assert all(section["text"] for section in body["sections"])

    def test_document_before_parsing_returns_404(self, client: TestClient, db_session, tmp_path) -> None:
        paper = _prepare_downloaded_paper(db_session, tmp_path, source_id="2401.88812")

        response = client.get(f"/api/v1/papers/{paper.paper_id}/document")

        assert response.status_code == 404

    def test_health_still_works_alongside_parsing_routes(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
