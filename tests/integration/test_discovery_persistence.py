"""Integration tests: search result -> normalized Paper -> Postgres upsert.

Requires a reachable database (see `tests/integration/conftest.py`);
skipped automatically otherwise.
"""

from datetime import datetime, timezone

from sqlalchemy import select

from app.core.config import Settings
from app.ingestion.discovery.models import ArxivPaperResult, PaperSearchQuery
from app.ingestion.discovery.service import PaperDiscoveryService
from app.storage.postgres.models import IngestionJobRecord
from app.storage.postgres.repositories.papers import PaperRepository


class StubArxivClient:
    def __init__(self, results: list[ArxivPaperResult]) -> None:
        self.results = results

    def search(self, query: PaperSearchQuery) -> list[ArxivPaperResult]:
        return self.results


def _result(**overrides) -> ArxivPaperResult:
    defaults = dict(
        source_id="2401.54321",
        version="v1",
        title="Original Title",
        abstract="An abstract.",
        authors=["Alice Smith"],
        categories=["cs.AI"],
        published_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return ArxivPaperResult(**defaults)


def _service(db_session, results: list[ArxivPaperResult]) -> PaperDiscoveryService:
    settings = Settings(ARXIV_DEFAULT_MAX_RESULTS=20, ARXIV_MAX_RESULTS_LIMIT=100)
    client = StubArxivClient(results)
    repository = PaperRepository(db_session)
    return PaperDiscoveryService(settings, client, repository)


class TestDiscoveryPersistence:
    def test_first_discovery_creates_a_row(self, db_session) -> None:
        service = _service(db_session, [_result()])

        results = service.search(PaperSearchQuery(query="graph rag"))

        assert len(results) == 1
        stored = PaperRepository(db_session).get_by_source("arxiv", "2401.54321")
        assert stored is not None
        assert results[0].already_known is False

    def test_second_discovery_does_not_duplicate(self, db_session) -> None:
        service = _service(db_session, [_result()])
        service.search(PaperSearchQuery(query="graph rag"))

        second_results = service.search(PaperSearchQuery(query="graph rag"))

        assert second_results[0].already_known is True
        stored = PaperRepository(db_session).get_by_source("arxiv", "2401.54321")
        assert stored is not None

    def test_metadata_update_keeps_the_same_paper_id(self, db_session) -> None:
        service = _service(db_session, [_result(title="Original Title")])
        first = service.search(PaperSearchQuery(query="graph rag"))[0]

        service_v2 = _service(db_session, [_result(title="Revised Title")])
        second = service_v2.search(PaperSearchQuery(query="graph rag"))[0]

        assert first.paper.paper_id == second.paper.paper_id
        stored = PaperRepository(db_session).get_by_paper_id(first.paper.paper_id)
        assert stored.title == "Revised Title"

    def test_version_is_persisted_and_deduplicated(self, db_session) -> None:
        service = _service(db_session, [_result(version="v1")])
        first = service.search(PaperSearchQuery(query="graph rag"))[0]

        # Searching again with the same version must not duplicate the
        # paper_versions row.
        service_again = _service(db_session, [_result(version="v1")])
        service_again.search(PaperSearchQuery(query="graph rag"))

        repo = PaperRepository(db_session)
        version = repo.get_paper_version("paper-version:arxiv:2401.54321:v1")
        assert version is not None
        assert version.paper_id == first.paper.paper_id

    def test_a_new_version_is_a_separate_paper_version_row(self, db_session) -> None:
        service_v1 = _service(db_session, [_result(version="v1")])
        service_v1.search(PaperSearchQuery(query="graph rag"))

        service_v2 = _service(db_session, [_result(version="v2")])
        second = service_v2.search(PaperSearchQuery(query="graph rag"))[0]

        assert second.latest_version == "v2"
        repo = PaperRepository(db_session)
        v1 = repo.get_paper_version("paper-version:arxiv:2401.54321:v1")
        v2 = repo.get_paper_version("paper-version:arxiv:2401.54321:v2")
        assert v1 is not None
        assert v2 is not None
        assert v1.paper_id == v2.paper_id

    def test_discovery_never_creates_an_ingestion_job(self, db_session) -> None:
        service = _service(db_session, [_result()])
        service.search(PaperSearchQuery(query="graph rag"))

        rows = db_session.execute(select(IngestionJobRecord)).scalars().all()
        assert rows == []
