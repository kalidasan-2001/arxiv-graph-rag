"""Integration test for `SessionFactory`'s commit/rollback transaction boundary.

Unlike the repository tests, this exercises `SessionFactory` itself (which
`db_session` deliberately bypasses), so it manages its own cleanup.
"""

import pytest
from sqlalchemy import select

from app.core.config import Settings
from app.domain.papers import Paper
from app.storage.postgres.models import PaperRecord
from app.storage.postgres.repositories.papers import PaperRepository
from app.storage.postgres.session import SessionFactory
from tests.integration.conftest import _resolve_test_database_url


@pytest.fixture
def session_factory(pg_engine) -> SessionFactory:
    url = _resolve_test_database_url()
    if not url:
        pytest.skip("no DATABASE_URL/TEST_DATABASE_URL configured")
    return SessionFactory(settings=Settings(DATABASE_URL=url))


def test_successful_unit_of_work_commits(session_factory: SessionFactory) -> None:
    paper = Paper.create(source="arxiv", source_id="2401.55555", title="Committed Paper")
    with session_factory() as session:
        PaperRepository(session).upsert_paper(paper)

    with session_factory() as session:
        stored = session.execute(
            select(PaperRecord).where(PaperRecord.paper_id == paper.paper_id)
        ).scalar_one_or_none()
        assert stored is not None
        # Clean up -- this test uses SessionFactory's real commit, so it
        # must remove what it wrote instead of relying on rollback.
        session.delete(stored)


def test_exception_inside_unit_of_work_rolls_back(session_factory: SessionFactory) -> None:
    paper = Paper.create(source="arxiv", source_id="2401.66666", title="Rolled Back Paper")

    with pytest.raises(RuntimeError):
        with session_factory() as session:
            PaperRepository(session).upsert_paper(paper)
            raise RuntimeError("simulated failure before commit")

    with session_factory() as session:
        stored = session.execute(
            select(PaperRecord).where(PaperRecord.paper_id == paper.paper_id)
        ).scalar_one_or_none()
        assert stored is None
