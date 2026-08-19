"""Integration tests for `POST /api/v1/papers/{paper_id}/extract-graph`
and `GET /api/v1/papers/{paper_id}/graph-extraction`.

No live network, no real LLM -- `get_llm_provider` is overridden with a
deterministic fake, following the same `app.dependency_overrides` pattern
used for `get_embedding_provider` in `test_vector_index_api.py`. Requires
a reachable database; skipped automatically otherwise. No Qdrant needed.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.domain.papers import Paper, PaperVersion
from app.ingestion.checksums import sha256_file
from app.ingestion.download.storage import PaperStorage
from app.llm.provider import get_llm_provider
from app.main import app
from app.storage.postgres.repositories.papers import PaperRepository
from app.storage.postgres.session import get_db_session
from tests.llm.fakes import FakeLLMProvider
from tests.parsing.pdf_fixtures import make_scientific_paper_pdf_bytes


@pytest.fixture
def client(db_session, tmp_path, monkeypatch) -> Iterator[TestClient]:
    monkeypatch.setenv("PAPER_STORAGE_PATH", str(tmp_path))
    get_settings.cache_clear()

    fake_provider = FakeLLMProvider()

    def override_get_db_session():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_llm_provider] = lambda: fake_provider
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db_session, None)
        app.dependency_overrides.pop(get_llm_provider, None)
        get_settings.cache_clear()


def _prepare_chunked_paper(client: TestClient, db_session, tmp_path, source_id: str) -> Paper:
    papers = PaperRepository(db_session)
    paper = papers.upsert_paper(
        Paper.create(source="arxiv", source_id=source_id, title="A Paper", authors=["Alice Author"])
    )
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
        version.paper_version_id, checksum=sha256_file(final_path), storage_path=str(final_path),
        file_size_bytes=final_path.stat().st_size, downloaded_at=version.created_at,
    )

    assert client.post(f"/api/v1/papers/{paper.paper_id}/parse").status_code == 200
    assert client.post(f"/api/v1/papers/{paper.paper_id}/chunk").status_code == 200
    return paper


class TestExtractGraphEndpoint:
    def test_extract_graph_returns_counts_and_fingerprints(
        self, client: TestClient, db_session, tmp_path
    ) -> None:
        paper = _prepare_chunked_paper(client, db_session, tmp_path, "2401.55501")

        response = client.post(f"/api/v1/papers/{paper.paper_id}/extract-graph")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "graph_indexing"
        assert body["entity_count"] >= 1  # at least the PAPER entity
        assert body["extraction_version"] == "v1"
        assert body["extraction_config_fingerprint"]
        assert body["graph_extraction_generation_fingerprint"]
        assert body["artifact_reused"] is False

    def test_second_extract_graph_reports_reused(
        self, client: TestClient, db_session, tmp_path
    ) -> None:
        paper = _prepare_chunked_paper(client, db_session, tmp_path, "2401.55502")

        client.post(f"/api/v1/papers/{paper.paper_id}/extract-graph")
        response = client.post(f"/api/v1/papers/{paper.paper_id}/extract-graph")

        assert response.json()["artifact_reused"] is True

    def test_extract_graph_without_chunking_returns_404(
        self, client: TestClient, db_session, tmp_path
    ) -> None:
        papers = PaperRepository(db_session)
        paper = papers.upsert_paper(
            Paper.create(source="arxiv", source_id="2401.55503", title="Not Chunked")
        )
        papers.get_or_create_paper_version(PaperVersion.create(paper_id=paper.paper_id, version="v1"))

        response = client.post(f"/api/v1/papers/{paper.paper_id}/extract-graph")

        assert response.status_code == 404

    def test_extract_graph_for_unknown_paper_returns_404(self, client: TestClient) -> None:
        response = client.post("/api/v1/papers/paper:arxiv:9999.99996/extract-graph")
        assert response.status_code == 404


class TestGraphExtractionInspectionEndpoint:
    def test_returns_entities_and_relationships(
        self, client: TestClient, db_session, tmp_path
    ) -> None:
        paper = _prepare_chunked_paper(client, db_session, tmp_path, "2401.55510")
        client.post(f"/api/v1/papers/{paper.paper_id}/extract-graph")

        response = client.get(f"/api/v1/papers/{paper.paper_id}/graph-extraction")

        assert response.status_code == 200
        body = response.json()
        assert body["entity_count"] == len(body["entities"])
        assert body["relationship_count"] == len(body["relationships"])
        # The deterministic PAPER entity is always present, entity_id ==
        # paper_id (trusted identity, not a name-hash -- prompt #12).
        assert any(
            e["entity_type"] == "paper" and e["entity_id"] == paper.paper_id
            for e in body["entities"]
        )

    def test_before_extraction_returns_404(self, client: TestClient, db_session, tmp_path) -> None:
        paper = _prepare_chunked_paper(client, db_session, tmp_path, "2401.55511")

        response = client.get(f"/api/v1/papers/{paper.paper_id}/graph-extraction")

        assert response.status_code == 404

    def test_health_still_works_alongside_extraction_routes(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
