"""Integration tests for `POST /api/v1/papers/{paper_id}/chunk` and
`GET /api/v1/papers/{paper_id}/chunks`.

No live network -- the "parsed" precondition is reached via a real
`POST /parse` call against a locally-staged PDF (mirroring
`test_parse_api.py`). PostgreSQL is real (transaction-isolated via
`db_session`); requires a reachable database, skipped automatically
otherwise.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
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


def _prepare_downloaded_paper(db_session, tmp_path, source_id: str = "2401.99801") -> Paper:
    papers = PaperRepository(db_session)
    paper = papers.upsert_paper(Paper.create(source="arxiv", source_id=source_id, title="A Paper"))
    version = papers.get_or_create_paper_version(
        PaperVersion.create(paper_id=paper.paper_id, version="v1")
    )

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


def _prepare_parsed_paper(client: TestClient, db_session, tmp_path, source_id: str) -> Paper:
    paper = _prepare_downloaded_paper(db_session, tmp_path, source_id=source_id)
    response = client.post(f"/api/v1/papers/{paper.paper_id}/parse")
    assert response.status_code == 200
    return paper


class TestChunkEndpoint:
    def test_chunk_returns_chunked_status_and_count(self, client: TestClient, db_session, tmp_path) -> None:
        paper = _prepare_parsed_paper(client, db_session, tmp_path, "2401.99801")

        response = client.post(f"/api/v1/papers/{paper.paper_id}/chunk")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "chunked"
        assert body["chunk_count"] > 0
        assert body["chunking_version"] == "v1"
        assert body["chunk_config_fingerprint"]  # prompt 6.1: non-empty fingerprint
        assert body["chunk_reused"] is False

    def test_second_chunk_reports_reused(self, client: TestClient, db_session, tmp_path) -> None:
        paper = _prepare_parsed_paper(client, db_session, tmp_path, "2401.99802")

        client.post(f"/api/v1/papers/{paper.paper_id}/chunk")
        response = client.post(f"/api/v1/papers/{paper.paper_id}/chunk")

        assert response.json()["chunk_reused"] is True

    def test_chunk_without_parsing_returns_404(self, client: TestClient, db_session, tmp_path) -> None:
        paper = _prepare_downloaded_paper(db_session, tmp_path, source_id="2401.99803")

        response = client.post(f"/api/v1/papers/{paper.paper_id}/chunk")

        assert response.status_code == 404  # ParseArtifactNotFoundError's existing mapping

    def test_chunk_for_unknown_paper_returns_404(self, client: TestClient) -> None:
        response = client.post("/api/v1/papers/paper:arxiv:9999.99998/chunk")
        assert response.status_code == 404

    def test_chunk_size_change_triggers_a_real_rechunk_end_to_end(
        self, client: TestClient, db_session, tmp_path, monkeypatch
    ) -> None:
        """Prompt 6.1 regression, exercised through the real HTTP API (not
        just the service directly): changing `CHUNK_SIZE_TOKENS` with
        `CHUNKING_VERSION` unchanged must still force a real rechunk."""

        paper = _prepare_parsed_paper(client, db_session, tmp_path, "2401.99804")

        first = client.post(f"/api/v1/papers/{paper.paper_id}/chunk").json()
        assert first["chunk_reused"] is False

        monkeypatch.setenv("CHUNK_SIZE_TOKENS", "50")
        get_settings.cache_clear()
        try:
            second = client.post(f"/api/v1/papers/{paper.paper_id}/chunk").json()
        finally:
            get_settings.cache_clear()

        assert second["chunk_reused"] is False  # a genuine rechunk happened
        assert second["chunk_config_fingerprint"] != first["chunk_config_fingerprint"]


class TestChunksEndpoint:
    def test_chunks_lists_previews_without_text_by_default(
        self, client: TestClient, db_session, tmp_path
    ) -> None:
        paper = _prepare_parsed_paper(client, db_session, tmp_path, "2401.99810")
        client.post(f"/api/v1/papers/{paper.paper_id}/chunk")

        response = client.get(f"/api/v1/papers/{paper.paper_id}/chunks")

        assert response.status_code == 200
        body = response.json()
        assert body["total_count"] > 0
        assert len(body["chunks"]) == body["total_count"] or len(body["chunks"]) == 50
        first = body["chunks"][0]
        assert first["text"] is None
        assert len(first["text_preview"]) <= 200

    def test_chunks_includes_text_when_requested(
        self, client: TestClient, db_session, tmp_path
    ) -> None:
        paper = _prepare_parsed_paper(client, db_session, tmp_path, "2401.99811")
        client.post(f"/api/v1/papers/{paper.paper_id}/chunk")

        response = client.get(
            f"/api/v1/papers/{paper.paper_id}/chunks", params={"include_text": "true"}
        )

        body = response.json()
        assert all(chunk["text"] for chunk in body["chunks"])

    def test_chunks_can_be_filtered_by_section_type(
        self, client: TestClient, db_session, tmp_path
    ) -> None:
        paper = _prepare_parsed_paper(client, db_session, tmp_path, "2401.99812")
        client.post(f"/api/v1/papers/{paper.paper_id}/chunk")
        unfiltered = client.get(f"/api/v1/papers/{paper.paper_id}/chunks").json()
        section_type = unfiltered["chunks"][0]["section_type"]

        response = client.get(
            f"/api/v1/papers/{paper.paper_id}/chunks", params={"section_type": section_type}
        )

        body = response.json()
        assert all(chunk["section_type"] == section_type for chunk in body["chunks"])

    def test_chunks_respects_limit_and_offset(self, client: TestClient, db_session, tmp_path) -> None:
        paper = _prepare_parsed_paper(client, db_session, tmp_path, "2401.99813")
        client.post(f"/api/v1/papers/{paper.paper_id}/chunk")

        response = client.get(
            f"/api/v1/papers/{paper.paper_id}/chunks", params={"limit": 1, "offset": 0}
        )

        assert len(response.json()["chunks"]) == 1

    def test_chunks_before_chunking_returns_404(self, client: TestClient, db_session, tmp_path) -> None:
        paper = _prepare_parsed_paper(client, db_session, tmp_path, "2401.99814")

        response = client.get(f"/api/v1/papers/{paper.paper_id}/chunks")

        assert response.status_code == 404

    def test_health_still_works_alongside_chunking_routes(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
