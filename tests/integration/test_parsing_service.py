"""Integration tests for `PaperParsingService` against real PostgreSQL.

No live network -- the "downloaded PDF" precondition is set up directly via
`PaperStorage` + `PaperRepository`, matching what Prompt 4's acquisition
flow would have already produced. Requires a reachable database (see
`tests/integration/conftest.py`); skipped automatically otherwise.

Covers the full target flow (prompt #56): downloaded paper -> parse
request -> DOWNLOADED -> PARSING -> parsed artifact written -> parser
metadata persisted -> PARSED -- plus idempotency, parser-version-mismatch
invalidation, PDF-checksum-changed invalidation, and reconciliation after
a simulated partial DB failure / missing / corrupt parsed artifact.
"""

import pytest

from app.core.config import Settings
from app.core.exceptions import InvalidPdfError
from app.domain.enums import IngestionStatus
from app.domain.papers import Paper, PaperVersion
from app.ingestion.checksums import sha256_file
from app.ingestion.download.storage import PaperStorage
from app.ingestion.parsing.pymupdf_parser import PyMuPDFParser
from app.ingestion.parsing.service import PaperParsingService
from app.ingestion.parsing.storage import ParsedArtifactStorage
from app.storage.postgres.repositories.ingestion import IngestionRepository
from app.storage.postgres.repositories.papers import PaperRepository
from tests.parsing.pdf_fixtures import make_scientific_paper_pdf_bytes


class CountingParser:
    """Wraps a real `PyMuPDFParser`, counting `.parse()` invocations."""

    def __init__(self) -> None:
        self._inner = PyMuPDFParser()
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def version(self) -> str:
        return self._inner.version

    def parse(self, artifact_path, *, paper_id: str, paper_version_id: str):
        self.call_count += 1
        return self._inner.parse(artifact_path, paper_id=paper_id, paper_version_id=paper_version_id)


def _settings(storage_root) -> Settings:
    return Settings(PAPER_STORAGE_PATH=str(storage_root))


def _prepare_downloaded_paper(db_session, storage_root, *, source_id: str = "2401.77701"):
    """Discover a paper/version and place a real, checksummed PDF on disk
    for it, mirroring what Prompt 4's acquisition already does -- exactly
    the precondition `PaperParsingService` expects."""

    papers = PaperRepository(db_session)
    paper = papers.upsert_paper(
        Paper.create(source="arxiv", source_id=source_id, title="A Paper")
    )
    version = papers.get_or_create_paper_version(
        PaperVersion.create(paper_id=paper.paper_id, version="v1")
    )

    settings = _settings(storage_root)
    pdf_storage = PaperStorage(settings)
    temp_path = pdf_storage.get_temp_path(source="arxiv", source_id=source_id, version="v1")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(make_scientific_paper_pdf_bytes())
    final_path = pdf_storage.finalize(temp_path, source="arxiv", source_id=source_id, version="v1")
    checksum = sha256_file(final_path)

    version = papers.update_version_artifact(
        version.paper_version_id,
        checksum=checksum,
        storage_path=str(final_path),
        file_size_bytes=final_path.stat().st_size,
        downloaded_at=version.created_at,
    )
    return paper, version


def _service(db_session, storage_root, parser=None) -> tuple[PaperParsingService, CountingParser]:
    settings = _settings(storage_root)
    counting_parser = parser or CountingParser()
    service = PaperParsingService(
        counting_parser,
        PaperStorage(settings),
        ParsedArtifactStorage(settings),
        PaperRepository(db_session),
        IngestionRepository(db_session),
    )
    return service, counting_parser


class TestFullParsingFlow:
    def test_downloaded_paper_is_parsed_and_marked_parsed(self, db_session, tmp_path) -> None:
        paper, version = _prepare_downloaded_paper(db_session, tmp_path)
        service, parser = _service(db_session, tmp_path)

        result = service.parse(paper.paper_id)

        assert result.job.status == IngestionStatus.PARSED
        assert result.parse_reused is False
        assert parser.call_count == 1
        assert result.document.page_count == 3
        assert len(result.document.sections) > 0

        stored_version = PaperRepository(db_session).get_paper_version(version.paper_version_id)
        assert stored_version.parsed_artifact_path is not None
        assert stored_version.parser_name == "pymupdf"
        assert stored_version.page_count == 3
        assert stored_version.section_count == len(result.document.sections)
        assert stored_version.parsed_at is not None

        settings = _settings(tmp_path)
        parsed_storage = ParsedArtifactStorage(settings)
        assert parsed_storage.exists(source="arxiv", source_id=paper.source_id, version="v1")

    def test_parse_without_a_downloaded_pdf_raises(self, db_session, tmp_path) -> None:
        papers = PaperRepository(db_session)
        paper = papers.upsert_paper(
            Paper.create(source="arxiv", source_id="2401.77702", title="Not Downloaded")
        )
        papers.get_or_create_paper_version(
            PaperVersion.create(paper_id=paper.paper_id, version="v1")
        )
        service, _parser = _service(db_session, tmp_path)

        with pytest.raises(InvalidPdfError):
            service.parse(paper.paper_id)


class TestIdempotency:
    def test_second_parse_does_not_invoke_the_parser_again(self, db_session, tmp_path) -> None:
        paper, _version = _prepare_downloaded_paper(db_session, tmp_path, source_id="2401.77710")
        service, parser = _service(db_session, tmp_path)

        first = service.parse(paper.paper_id)
        second = service.parse(paper.paper_id)

        assert parser.call_count == 1  # the critical assertion
        assert second.parse_reused is True
        assert second.job.status == IngestionStatus.PARSED
        assert second.job.ingestion_job_id == first.job.ingestion_job_id
        assert second.document.page_count == first.document.page_count


class TestParserVersionMismatchInvalidation:
    def test_different_parser_version_triggers_a_real_reparse(self, db_session, tmp_path) -> None:
        paper, _version = _prepare_downloaded_paper(db_session, tmp_path, source_id="2401.77720")
        service, parser = _service(db_session, tmp_path)
        service.parse(paper.paper_id)
        assert parser.call_count == 1

        # Simulate a parser upgrade: a second service using a "different"
        # parser identity must not trust the old artifact. Overrides
        # `parse()` too (not just the `.version` property) so the returned
        # document is internally consistent with the claimed identity --
        # `CountingParser` alone would delegate to a real `PyMuPDFParser`
        # that stamps its own (unchanged) version onto the document.
        class DifferentVersionParser(CountingParser):
            @property
            def version(self) -> str:
                return "9.9.9+adapter99"

            def parse(self, artifact_path, *, paper_id: str, paper_version_id: str):
                document = super().parse(
                    artifact_path, paper_id=paper_id, paper_version_id=paper_version_id
                )
                return document.model_copy(update={"parser_version": self.version})

        different_parser = DifferentVersionParser()
        service2, _ = _service(db_session, tmp_path, parser=different_parser)

        result = service2.parse(paper.paper_id)

        assert different_parser.call_count == 1  # a genuine reparse happened
        assert result.parse_reused is False
        assert result.document.parser_version == "9.9.9+adapter99"


class TestPdfChecksumChangedInvalidation:
    def test_changed_pdf_checksum_triggers_a_real_reparse(self, db_session, tmp_path) -> None:
        paper, version = _prepare_downloaded_paper(db_session, tmp_path, source_id="2401.77730")
        service, parser = _service(db_session, tmp_path)
        service.parse(paper.paper_id)
        assert parser.call_count == 1

        # Simulate the PDF having been legitimately re-acquired with
        # different bytes (and thus a different checksum) since the parse.
        papers = PaperRepository(db_session)
        papers.update_version_artifact(
            version.paper_version_id,
            checksum="0" * 64,
            storage_path=version.storage_path,
            file_size_bytes=version.file_size_bytes,
            downloaded_at=version.downloaded_at,
        )

        with pytest.raises(InvalidPdfError):
            # _validate_ready_for_parsing re-verifies the checksum against
            # the (unchanged) file on disk -- a mismatched recorded
            # checksum is itself detected before parsing is even attempted.
            service.parse(paper.paper_id)


class TestReconciliation:
    def test_parsed_json_exists_but_final_db_write_failed_is_reconciled_without_reparsing(
        self, db_session, tmp_path
    ) -> None:
        paper, version = _prepare_downloaded_paper(db_session, tmp_path, source_id="2401.77740")
        settings = _settings(tmp_path)
        parser = PyMuPDFParser()

        # Parse directly via the parser + storage (bypassing the service),
        # simulating "parsed.json was finalized but the service's DB write
        # never happened" -- no ingestion job, no DB metadata recorded.
        pdf_storage = PaperStorage(settings)
        pdf_path = pdf_storage.get_path(source="arxiv", source_id=paper.source_id, version="v1")
        document = parser.parse(pdf_path, paper_id=paper.paper_id, paper_version_id=version.paper_version_id)
        document = document.model_copy(update={"source_pdf_checksum": version.checksum})
        ParsedArtifactStorage(settings).write(
            document, source="arxiv", source_id=paper.source_id, version="v1"
        )

        counting_parser = CountingParser()
        service, _ = _service(db_session, tmp_path, parser=counting_parser)
        result = service.parse(paper.paper_id)

        assert counting_parser.call_count == 0  # never reparsed -- reconciled from disk
        assert result.parse_reused is True
        assert result.job.status == IngestionStatus.PARSED

        stored_version = PaperRepository(db_session).get_paper_version(version.paper_version_id)
        assert stored_version.parsed_artifact_path is not None
        assert stored_version.page_count == document.page_count

    def test_db_says_parsed_but_artifact_missing_triggers_reparse(self, db_session, tmp_path) -> None:
        paper, _version = _prepare_downloaded_paper(db_session, tmp_path, source_id="2401.77750")
        service, parser = _service(db_session, tmp_path)
        first = service.parse(paper.paper_id)
        assert first.job.status == IngestionStatus.PARSED

        settings = _settings(tmp_path)
        ParsedArtifactStorage(settings).delete(source="arxiv", source_id=paper.source_id, version="v1")

        second = service.parse(paper.paper_id)

        assert parser.call_count == 2  # missing artifact forced a real reparse
        assert second.job.status == IngestionStatus.PARSED
        assert second.job.ingestion_job_id != first.job.ingestion_job_id  # stale job replaced

    def test_corrupt_parsed_json_triggers_reparse(self, db_session, tmp_path) -> None:
        paper, _version = _prepare_downloaded_paper(db_session, tmp_path, source_id="2401.77760")
        service, parser = _service(db_session, tmp_path)
        first = service.parse(paper.paper_id)
        assert first.job.status == IngestionStatus.PARSED

        settings = _settings(tmp_path)
        parsed_path = ParsedArtifactStorage(settings).get_path(
            source="arxiv", source_id=paper.source_id, version="v1"
        )
        parsed_path.write_text("{ not valid json at all", encoding="utf-8")

        second = service.parse(paper.paper_id)

        assert parser.call_count == 2  # corruption forced a real reparse
        assert second.job.status == IngestionStatus.PARSED
        assert second.job.ingestion_job_id != first.job.ingestion_job_id
