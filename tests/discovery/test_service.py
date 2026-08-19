"""Unit tests for `PaperDiscoveryService` using a stub `ArxivClient`.

`paper_repository=None` throughout -- persistence behavior is covered
separately by `tests/integration/test_discovery_persistence.py` against a
real PostgreSQL instance.
"""

from datetime import datetime, timezone

from app.core.config import Settings
from app.core.exceptions import InvalidSearchQueryError
from app.ingestion.discovery.models import ArxivPaperResult, PaperSearchQuery
from app.ingestion.discovery.service import PaperDiscoveryService
import pytest


class StubArxivClient:
    """Records the query it was called with and returns a canned list."""

    def __init__(self, results: list[ArxivPaperResult]) -> None:
        self.results = results
        self.last_query: PaperSearchQuery | None = None

    def search(self, query: PaperSearchQuery) -> list[ArxivPaperResult]:
        self.last_query = query
        return self.results


def _result(**overrides) -> ArxivPaperResult:
    defaults = dict(
        source_id="2401.12345",
        version="v1",
        title="A Paper",
        abstract="An abstract.",
        authors=["Alice Smith"],
        categories=["cs.AI"],
        published_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return ArxivPaperResult(**defaults)


def _service(client: StubArxivClient, **settings_overrides) -> PaperDiscoveryService:
    defaults = dict(ARXIV_DEFAULT_MAX_RESULTS=20, ARXIV_MAX_RESULTS_LIMIT=100)
    defaults.update(settings_overrides)
    return PaperDiscoveryService(Settings(**defaults), client, paper_repository=None)


class TestSearchReturnsNormalizedResults:
    def test_returns_domain_papers_without_persistence(self) -> None:
        client = StubArxivClient([_result()])
        service = _service(client)

        results = service.search(PaperSearchQuery(query="graph rag"))

        assert len(results) == 1
        assert results[0].paper.paper_id == "paper:arxiv:2401.12345"
        assert results[0].already_known is False

    def test_resolves_default_max_results_onto_the_client_query(self) -> None:
        client = StubArxivClient([])
        service = _service(client, ARXIV_DEFAULT_MAX_RESULTS=7)

        service.search(PaperSearchQuery(query="graph rag"))

        assert client.last_query is not None
        assert client.last_query.max_results == 7

    def test_over_limit_max_results_raises_before_calling_the_client(self) -> None:
        client = StubArxivClient([])
        service = _service(client, ARXIV_MAX_RESULTS_LIMIT=50)

        with pytest.raises(InvalidSearchQueryError):
            service.search(PaperSearchQuery(query="graph rag", max_results=500))

        assert client.last_query is None


class TestDeduplication:
    def test_repeated_source_id_collapses_to_one_result(self) -> None:
        client = StubArxivClient([_result(version="v1"), _result(version="v1")])
        service = _service(client)

        results = service.search(PaperSearchQuery(query="graph rag"))

        assert len(results) == 1

    def test_keeps_the_highest_version_for_a_repeated_source_id(self) -> None:
        client = StubArxivClient(
            [_result(version="v1"), _result(version="v3"), _result(version="v2")]
        )
        service = _service(client)

        results = service.search(PaperSearchQuery(query="graph rag"))

        assert len(results) == 1
        assert results[0].latest_version == "v3"

    def test_different_source_ids_are_not_merged(self) -> None:
        client = StubArxivClient([_result(source_id="2401.11111"), _result(source_id="2401.22222")])
        service = _service(client)

        results = service.search(PaperSearchQuery(query="graph rag"))

        assert {r.paper.source_id for r in results} == {"2401.11111", "2401.22222"}


class TestDateRangeFiltering:
    def test_filters_out_results_before_published_after(self) -> None:
        client = StubArxivClient(
            [
                _result(source_id="2401.11111", published_at=datetime(2023, 1, 1, tzinfo=timezone.utc)),
                _result(source_id="2401.22222", published_at=datetime(2024, 6, 1, tzinfo=timezone.utc)),
            ]
        )
        service = _service(client)

        results = service.search(
            PaperSearchQuery(
                query="graph rag", published_after=datetime(2024, 1, 1, tzinfo=timezone.utc)
            )
        )

        assert [r.paper.source_id for r in results] == ["2401.22222"]

    def test_filters_out_results_after_published_before(self) -> None:
        client = StubArxivClient(
            [
                _result(source_id="2401.11111", published_at=datetime(2023, 1, 1, tzinfo=timezone.utc)),
                _result(source_id="2401.22222", published_at=datetime(2024, 6, 1, tzinfo=timezone.utc)),
            ]
        )
        service = _service(client)

        results = service.search(
            PaperSearchQuery(
                query="graph rag", published_before=datetime(2024, 1, 1, tzinfo=timezone.utc)
            )
        )

        assert [r.paper.source_id for r in results] == ["2401.11111"]

    def test_no_date_filter_returns_everything(self) -> None:
        client = StubArxivClient([_result(source_id="2401.11111"), _result(source_id="2401.22222")])
        service = _service(client)

        results = service.search(PaperSearchQuery(query="graph rag"))

        assert len(results) == 2
