"""Integration tests for `PaperRepository` against real PostgreSQL.

Requires a reachable database (see `tests/integration/conftest.py`);
skipped automatically otherwise.
"""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.exceptions import PersistenceConflictError
from app.domain.papers import Paper, PaperVersion
from app.storage.postgres.models import PaperRecord
from app.storage.postgres.repositories.papers import PaperRepository


def _paper(**overrides) -> Paper:
    defaults = {"source": "arxiv", "source_id": "2401.12345", "title": "Original Title"}
    defaults.update(overrides)
    return Paper.create(**defaults)


class TestPaperUpsert:
    def test_first_upsert_inserts_one_row(self, db_session) -> None:
        PaperRepository(db_session).upsert_paper(_paper())

        rows = db_session.execute(select(PaperRecord)).scalars().all()
        assert len(rows) == 1

    def test_second_identical_upsert_does_not_duplicate(self, db_session) -> None:
        repo = PaperRepository(db_session)
        repo.upsert_paper(_paper())
        repo.upsert_paper(_paper())

        rows = db_session.execute(select(PaperRecord)).scalars().all()
        assert len(rows) == 1

    def test_upsert_updates_metadata_but_keeps_same_paper_id(self, db_session) -> None:
        repo = PaperRepository(db_session)
        first = repo.upsert_paper(_paper(title="Original Title"))
        second = repo.upsert_paper(_paper(title="Revised Title"))

        assert first.paper_id == second.paper_id
        assert repo.get_by_paper_id(first.paper_id).title == "Revised Title"

    def test_duplicate_source_and_source_id_violates_unique_constraint(self, db_session) -> None:
        # Simulate a race that bypasses the repository's own idempotency
        # check -- the database constraint must still refuse a second
        # logical paper record for the same (source, source_id).
        paper_a = _paper()
        db_session.add(
            PaperRecord(
                paper_id=paper_a.paper_id,
                source=paper_a.source,
                source_id=paper_a.source_id,
                title="A",
                authors=[],
                categories=[],
            )
        )
        db_session.flush()

        db_session.add(
            PaperRecord(
                paper_id=paper_a.paper_id + ":duplicate-race",
                source=paper_a.source,
                source_id=paper_a.source_id,
                title="B",
                authors=[],
                categories=[],
            )
        )
        with pytest.raises(IntegrityError):
            db_session.flush()


class TestPaperVersions:
    def test_creating_v1_persists_it(self, db_session) -> None:
        repo = PaperRepository(db_session)
        paper = repo.upsert_paper(_paper())

        version = repo.get_or_create_paper_version(
            PaperVersion.create(paper_id=paper.paper_id, version="1")
        )

        assert version.version == "1"
        assert repo.get_paper_version(version.paper_version_id) is not None

    def test_requesting_v1_again_reuses_it_without_duplicating(self, db_session) -> None:
        repo = PaperRepository(db_session)
        paper = repo.upsert_paper(_paper())

        first = repo.get_or_create_paper_version(
            PaperVersion.create(paper_id=paper.paper_id, version="1")
        )
        second = repo.get_or_create_paper_version(
            PaperVersion.create(paper_id=paper.paper_id, version="1")
        )

        assert first.paper_version_id == second.paper_version_id

    def test_v2_is_a_separate_version_of_the_same_logical_paper(self, db_session) -> None:
        repo = PaperRepository(db_session)
        paper = repo.upsert_paper(_paper())

        v1 = repo.get_or_create_paper_version(
            PaperVersion.create(paper_id=paper.paper_id, version="1")
        )
        v2 = repo.get_or_create_paper_version(
            PaperVersion.create(paper_id=paper.paper_id, version="2")
        )

        assert v1.paper_version_id != v2.paper_version_id
        assert v1.paper_id == v2.paper_id == paper.paper_id

    def test_conflicting_checksum_on_existing_version_is_rejected(self, db_session) -> None:
        repo = PaperRepository(db_session)
        paper = repo.upsert_paper(_paper())
        repo.get_or_create_paper_version(
            PaperVersion.create(paper_id=paper.paper_id, version="1", checksum="aaa")
        )

        with pytest.raises(PersistenceConflictError):
            repo.get_or_create_paper_version(
                PaperVersion.create(paper_id=paper.paper_id, version="1", checksum="bbb")
            )
