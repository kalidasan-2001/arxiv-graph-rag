"""Integration test for `GET /api/v1/papers/search`.

Uses FastAPI dependency overrides: `get_arxiv_client` is replaced with a
fake (no real network), `get_db_session` is replaced with the
transaction-isolated `db_session` fixture (real PostgreSQL, auto-rolled
back). Requires a reachable database; skipped automatically otherwise.
"""

from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.ingestion.discovery.arxiv_client import get_arxiv_client
from app.ingestion.discovery.models import ArxivPaperResult, PaperSearchQuery
from app.main import app
from app.storage.postgres.session import get_db_session


class FakeArxivClient:
    def search(self, query: PaperSearchQuery) -> list[ArxivPaperResult]:
        return [
            ArxivPaperResult(
                source_id="2401.99999",
                version="v1",
                title="Graph RAG: A Survey",
                abstract="An abstract about hybrid retrieval.",
                authors=["Alice Smith"],
                categories=["cs.AI"],
                published_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                pdf_url="http://arxiv.org/pdf/2401.99999v1",
            )
        ]


@pytest.fixture
def client(db_session) -> Iterator[TestClient]:
    def override_get_db_session():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_arxiv_client] = lambda: FakeArxivClient()
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db_session, None)
        app.dependency_overrides.pop(get_arxiv_client, None)


class TestPapersSearchApi:
    def test_search_returns_normalized_results(self, client: TestClient) -> None:
        response = client.get("/api/v1/papers/search", params={"q": "graph rag"})

        assert response.status_code == 200
        body = response.json()
        assert body["query"] == "graph rag"
        assert body["count"] == 1
        result = body["results"][0]
        assert result["paper_id"] == "paper:arxiv:2401.99999"
        assert result["source_id"] == "2401.99999"
        assert result["latest_version"] == "v1"
        assert result["title"] == "Graph RAG: A Survey"
        assert result["already_known"] is False

    def test_second_search_reports_already_known(self, client: TestClient) -> None:
        client.get("/api/v1/papers/search", params={"q": "graph rag"})

        response = client.get("/api/v1/papers/search", params={"q": "graph rag"})

        assert response.json()["results"][0]["already_known"] is True

    def test_blank_query_returns_400(self, client: TestClient) -> None:
        response = client.get("/api/v1/papers/search", params={"q": ""})
        assert response.status_code in (400, 422)

    def test_max_results_over_limit_returns_400(self, client: TestClient) -> None:
        response = client.get(
            "/api/v1/papers/search", params={"q": "graph rag", "max_results": 100000}
        )
        assert response.status_code == 400

    def test_health_still_works_alongside_papers_route(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
