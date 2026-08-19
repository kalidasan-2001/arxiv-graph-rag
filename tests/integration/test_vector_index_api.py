"""Integration tests for `POST /api/v1/papers/{paper_id}/vector-index`.

No live network, no real embedding model -- `get_embedding_provider` and
`get_vector_repository` are overridden with a deterministic fake provider
and a real (but disposable, uniquely-named) Qdrant collection, following
the same `app.dependency_overrides` pattern already used for `get_db_session`
in `test_chunk_api.py`. Requires both a reachable database and a reachable
Qdrant; skipped automatically otherwise.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.domain.papers import Paper, PaperVersion
from app.embeddings.provider import get_embedding_provider
from app.ingestion.checksums import sha256_file
from app.ingestion.download.storage import PaperStorage
from app.main import app
from app.storage.postgres.repositories.papers import PaperRepository
from app.storage.postgres.session import get_db_session
from app.storage.qdrant.qdrant_repository import QdrantVectorRepository, get_vector_repository
from tests.embeddings.fakes import FakeEmbeddingProvider
from tests.parsing.pdf_fixtures import make_scientific_paper_pdf_bytes


@pytest.fixture
def client(
    db_session, tmp_path, monkeypatch, qdrant_client, qdrant_collection_name
) -> Iterator[TestClient]:
    monkeypatch.setenv("PAPER_STORAGE_PATH", str(tmp_path))
    get_settings.cache_clear()

    fake_provider = FakeEmbeddingProvider(dimension=8, normalize=True)
    fake_repo = QdrantVectorRepository(qdrant_client, qdrant_collection_name)

    def override_get_db_session():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_embedding_provider] = lambda: fake_provider
    app.dependency_overrides[get_vector_repository] = lambda: fake_repo
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db_session, None)
        app.dependency_overrides.pop(get_embedding_provider, None)
        app.dependency_overrides.pop(get_vector_repository, None)
        get_settings.cache_clear()


def _prepare_chunked_paper(client: TestClient, db_session, tmp_path, source_id: str) -> Paper:
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
        version.paper_version_id, checksum=sha256_file(final_path), storage_path=str(final_path),
        file_size_bytes=final_path.stat().st_size, downloaded_at=version.created_at,
    )

    assert client.post(f"/api/v1/papers/{paper.paper_id}/parse").status_code == 200
    assert client.post(f"/api/v1/papers/{paper.paper_id}/chunk").status_code == 200
    return paper


class TestVectorIndexEndpoint:
    def test_vector_index_returns_indexed_status_and_count(
        self, client: TestClient, db_session, tmp_path
    ) -> None:
        paper = _prepare_chunked_paper(client, db_session, tmp_path, "2401.66601")

        response = client.post(f"/api/v1/papers/{paper.paper_id}/vector-index")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "vector_indexed"
        assert body["vector_count"] > 0
        assert body["embedding_provider"] == "fake"
        assert body["embedding_config_fingerprint"]
        assert body["vector_generation_fingerprint"]
        assert body["artifact_reused"] is False

    def test_second_vector_index_reports_reused(
        self, client: TestClient, db_session, tmp_path
    ) -> None:
        paper = _prepare_chunked_paper(client, db_session, tmp_path, "2401.66602")

        client.post(f"/api/v1/papers/{paper.paper_id}/vector-index")
        response = client.post(f"/api/v1/papers/{paper.paper_id}/vector-index")

        assert response.json()["artifact_reused"] is True

    def test_vector_index_without_chunking_returns_404(
        self, client: TestClient, db_session, tmp_path
    ) -> None:
        papers = PaperRepository(db_session)
        paper = papers.upsert_paper(
            Paper.create(source="arxiv", source_id="2401.66603", title="Not Chunked")
        )
        papers.get_or_create_paper_version(PaperVersion.create(paper_id=paper.paper_id, version="v1"))

        response = client.post(f"/api/v1/papers/{paper.paper_id}/vector-index")

        assert response.status_code == 404  # ChunkArtifactNotFoundError's existing mapping

    def test_vector_index_for_unknown_paper_returns_404(self, client: TestClient) -> None:
        response = client.post("/api/v1/papers/paper:arxiv:9999.99997/vector-index")
        assert response.status_code == 404

    def test_health_still_works_alongside_vector_index_routes(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
