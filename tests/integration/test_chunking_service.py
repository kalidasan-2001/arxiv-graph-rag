"""Integration tests for `ChunkingService` against real PostgreSQL.

No live network -- the "parsed" precondition is set up directly via a real
`PyMuPDFParser` + `ParsedArtifactStorage`, matching what Prompt 5's parsing
flow would have already produced. Requires a reachable database (see
`tests/integration/conftest.py`); skipped automatically otherwise.

Covers the full target flow: parsed paper -> chunk request -> PARSED ->
CHUNKING -> chunk artifact written -> chunk metadata persisted -> CHUNKED
-- plus idempotency, chunk-size/overlap/chunking-version-changed
invalidation, parsed-artifact-checksum-changed invalidation, and
reconciliation after a simulated partial DB write failure / missing /
corrupt chunk artifact.
"""

import json

import pytest

from app.core.config import Settings
from app.core.exceptions import ParseArtifactNotFoundError
from app.domain.enums import IngestionStatus
from app.domain.papers import Paper, PaperVersion
from app.ingestion.checksums import sha256_file
from app.ingestion.chunking.section_chunker import SectionAwareChunker
from app.ingestion.chunking.service import ChunkingService
from app.ingestion.chunking.storage import ChunkArtifactStorage
from app.ingestion.download.storage import PaperStorage
from app.ingestion.parsing.pymupdf_parser import PyMuPDFParser
from app.ingestion.parsing.storage import ParsedArtifactStorage
from app.storage.postgres.repositories.ingestion import IngestionRepository
from app.storage.postgres.repositories.papers import PaperRepository
from tests.parsing.pdf_fixtures import make_scientific_paper_pdf_bytes


class CountingChunker:
    """Wraps a real `SectionAwareChunker`, counting `.chunk()` invocations."""

    def __init__(self, settings: Settings, token_counter=None) -> None:
        self._inner = SectionAwareChunker(settings, token_counter)
        self.call_count = 0

    @property
    def chunking_version(self) -> str:
        return self._inner.chunking_version

    @property
    def config_fingerprint(self) -> str:
        return self._inner.config_fingerprint

    def chunk(self, document):
        self.call_count += 1
        return self._inner.chunk(document)


class _AlternateTokenCounter:
    """A stand-in "different tokenizer" -- same counting behavior as
    `WhitespaceTokenCounter`, different identity (prompt 6.1 tokenizer
    invalidation test)."""

    @property
    def name(self) -> str:
        return "alternate-tokenizer-v1"

    @property
    def version(self) -> str | None:
        return "1.2.3"

    def count(self, text: str) -> int:
        return len(text.split())


def _settings(storage_root, **overrides) -> Settings:
    return Settings(PAPER_STORAGE_PATH=str(storage_root), **overrides)


def _prepare_parsed_paper(db_session, storage_root, *, source_id: str = "2401.88801"):
    """Discover a paper/version, place a real PDF on disk, and parse it for
    real -- exactly the precondition `ChunkingService` expects."""

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
    pdf_checksum = sha256_file(final_path)

    version = papers.update_version_artifact(
        version.paper_version_id,
        checksum=pdf_checksum,
        storage_path=str(final_path),
        file_size_bytes=final_path.stat().st_size,
        downloaded_at=version.created_at,
    )

    parser = PyMuPDFParser()
    document = parser.parse(final_path, paper_id=paper.paper_id, paper_version_id=version.paper_version_id)
    document = document.model_copy(update={"source_pdf_checksum": pdf_checksum})
    parsed_storage = ParsedArtifactStorage(settings)
    parsed_path = parsed_storage.write(document, source="arxiv", source_id=source_id, version="v1")
    parsed_checksum = sha256_file(parsed_path)

    version = papers.update_version_parse_result(
        version.paper_version_id,
        parsed_artifact_path=str(parsed_path),
        parsed_at=version.created_at,
        parser_name=document.parser_name,
        parser_version=document.parser_version,
        page_count=document.page_count,
        section_count=len(document.sections),
        warning_count=len(document.warnings),
    )
    return paper, version, parsed_checksum


def _service(
    db_session, storage_root, chunker=None, token_counter=None, **settings_overrides
) -> tuple[ChunkingService, CountingChunker]:
    settings = _settings(storage_root, **settings_overrides)
    counting_chunker = chunker or CountingChunker(settings, token_counter)
    service = ChunkingService(
        counting_chunker,
        ParsedArtifactStorage(settings),
        ChunkArtifactStorage(settings),
        PaperRepository(db_session),
        IngestionRepository(db_session),
    )
    return service, counting_chunker


class TestFullChunkingFlow:
    def test_parsed_paper_is_chunked_and_marked_chunked(self, db_session, tmp_path) -> None:
        paper, version, _parsed_checksum = _prepare_parsed_paper(db_session, tmp_path)
        service, chunker = _service(db_session, tmp_path)

        result = service.chunk(paper.paper_id)

        assert result.job.status == IngestionStatus.CHUNKED
        assert result.chunk_reused is False
        assert chunker.call_count == 1
        assert len(result.document.chunks) > 0

        stored_version = PaperRepository(db_session).get_paper_version(version.paper_version_id)
        assert stored_version.chunked_artifact_path is not None
        assert stored_version.chunk_count == len(result.document.chunks)
        assert stored_version.chunking_version == chunker.chunking_version
        assert stored_version.chunked_at is not None
        assert stored_version.chunk_artifact_checksum is not None

        settings = _settings(tmp_path)
        chunk_storage = ChunkArtifactStorage(settings)
        assert chunk_storage.exists(source="arxiv", source_id=paper.source_id, version="v1")

    def test_chunk_without_a_parsed_artifact_raises(self, db_session, tmp_path) -> None:
        papers = PaperRepository(db_session)
        paper = papers.upsert_paper(
            Paper.create(source="arxiv", source_id="2401.88802", title="Not Parsed")
        )
        papers.get_or_create_paper_version(
            PaperVersion.create(paper_id=paper.paper_id, version="v1")
        )
        service, _chunker = _service(db_session, tmp_path)

        with pytest.raises(ParseArtifactNotFoundError):
            service.chunk(paper.paper_id)


class TestIdempotency:
    def test_second_chunk_does_not_invoke_the_chunker_again(self, db_session, tmp_path) -> None:
        paper, _version, _checksum = _prepare_parsed_paper(db_session, tmp_path, source_id="2401.88810")
        service, chunker = _service(db_session, tmp_path)

        first = service.chunk(paper.paper_id)
        second = service.chunk(paper.paper_id)

        assert chunker.call_count == 1  # the critical assertion
        assert second.chunk_reused is True
        assert second.job.status == IngestionStatus.CHUNKED
        assert second.job.ingestion_job_id == first.job.ingestion_job_id
        assert len(second.document.chunks) == len(first.document.chunks)


class TestChunkSizeChangedInvalidation:
    def test_different_chunk_size_triggers_a_real_rechunk(self, db_session, tmp_path) -> None:
        # Prompt 6.1: this used to reuse the stale artifact because only
        # `chunking.version` was compared, and `CHUNK_SIZE_TOKENS` alone
        # doesn't change it -- that was exactly the unsafe gap this stage
        # closes. Reuse condition now compares `config_fingerprint`, which
        # is sensitive to chunk size even with an unchanged version.
        paper, version, _checksum = _prepare_parsed_paper(db_session, tmp_path, source_id="2401.88820")
        service, chunker = _service(db_session, tmp_path, CHUNK_SIZE_TOKENS=700)
        service.chunk(paper.paper_id)
        assert chunker.call_count == 1

        service2, chunker2 = _service(db_session, tmp_path, CHUNK_SIZE_TOKENS=50)
        result = service2.chunk(paper.paper_id)

        assert chunker2.call_count == 1  # a genuine rechunk happened
        assert result.chunk_reused is False
        assert result.document.chunking.chunk_size_tokens == 50

        stored_version = PaperRepository(db_session).get_paper_version(version.paper_version_id)
        assert stored_version.chunk_config_fingerprint == result.document.chunking.config_fingerprint


class TestOverlapChangedInvalidation:
    def test_different_overlap_triggers_a_real_rechunk(self, db_session, tmp_path) -> None:
        paper, _version, _checksum = _prepare_parsed_paper(db_session, tmp_path, source_id="2401.88821")
        service, chunker = _service(db_session, tmp_path, CHUNK_OVERLAP_TOKENS=100)
        service.chunk(paper.paper_id)
        assert chunker.call_count == 1

        service2, chunker2 = _service(db_session, tmp_path, CHUNK_OVERLAP_TOKENS=150)
        result = service2.chunk(paper.paper_id)

        assert chunker2.call_count == 1
        assert result.chunk_reused is False
        assert result.document.chunking.chunk_overlap_tokens == 150


class TestMinChunkTokensChangedInvalidation:
    def test_different_min_chunk_tokens_triggers_a_real_rechunk(self, db_session, tmp_path) -> None:
        paper, _version, _checksum = _prepare_parsed_paper(db_session, tmp_path, source_id="2401.88822")
        service, chunker = _service(db_session, tmp_path, MIN_CHUNK_TOKENS=80)
        service.chunk(paper.paper_id)
        assert chunker.call_count == 1

        service2, chunker2 = _service(db_session, tmp_path, MIN_CHUNK_TOKENS=120)
        result = service2.chunk(paper.paper_id)

        assert chunker2.call_count == 1
        assert result.chunk_reused is False
        assert result.document.chunking.min_chunk_tokens == 120


class TestTokenizerChangedInvalidation:
    def test_different_tokenizer_triggers_a_real_rechunk(self, db_session, tmp_path) -> None:
        paper, _version, _checksum = _prepare_parsed_paper(db_session, tmp_path, source_id="2401.88823")
        service, chunker = _service(db_session, tmp_path)
        service.chunk(paper.paper_id)
        assert chunker.call_count == 1

        service2, chunker2 = _service(db_session, tmp_path, token_counter=_AlternateTokenCounter())
        result = service2.chunk(paper.paper_id)

        assert chunker2.call_count == 1  # a genuine rechunk happened
        assert result.chunk_reused is False
        assert result.document.chunking.tokenizer == "alternate-tokenizer-v1"


class TestChunkingVersionChangedInvalidation:
    def test_different_chunking_version_triggers_a_real_rechunk(self, db_session, tmp_path) -> None:
        paper, _version, _checksum = _prepare_parsed_paper(db_session, tmp_path, source_id="2401.88830")
        service, chunker = _service(db_session, tmp_path, CHUNKING_VERSION="v1")
        service.chunk(paper.paper_id)
        assert chunker.call_count == 1

        service2, chunker2 = _service(db_session, tmp_path, CHUNKING_VERSION="v2")
        result = service2.chunk(paper.paper_id)

        assert chunker2.call_count == 1  # a genuine rechunk happened
        assert result.chunk_reused is False
        assert result.document.chunking.version == "v2"


class TestSameConfigurationReuse:
    def test_identical_effective_configuration_reuses_without_rechunking(
        self, db_session, tmp_path
    ) -> None:
        """Prompt 6.1's positive case: same parsed checksum, same chunk
        size, overlap, minimum size, tokenizer, and version -> reuse, the
        chunker is never invoked a second time."""

        paper, _version, _checksum = _prepare_parsed_paper(db_session, tmp_path, source_id="2401.88824")
        service, chunker = _service(
            db_session, tmp_path,
            CHUNKING_VERSION="v1", CHUNK_SIZE_TOKENS=700, CHUNK_OVERLAP_TOKENS=100,
            MIN_CHUNK_TOKENS=80,
        )
        service.chunk(paper.paper_id)
        assert chunker.call_count == 1

        service2, chunker2 = _service(
            db_session, tmp_path,
            CHUNKING_VERSION="v1", CHUNK_SIZE_TOKENS=700, CHUNK_OVERLAP_TOKENS=100,
            MIN_CHUNK_TOKENS=80,
        )
        result = service2.chunk(paper.paper_id)

        assert chunker2.call_count == 0
        assert result.chunk_reused is True


class TestLegacyArtifactHandledSafely:
    def test_chunks_json_without_config_fingerprint_is_treated_as_stale(
        self, db_session, tmp_path
    ) -> None:
        """Prompt 6.1 backward compatibility (prompt #9): a `chunks.json`
        written before `config_fingerprint` existed must never crash the
        service and must never be silently reused -- the next explicit
        chunk operation safely regenerates it."""

        paper, version, _parsed_checksum = _prepare_parsed_paper(
            db_session, tmp_path, source_id="2401.88825"
        )
        settings = _settings(tmp_path)
        chunk_storage = ChunkArtifactStorage(settings)
        legacy_path = chunk_storage.get_path(source="arxiv", source_id=paper.source_id, version="v1")
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_payload = {
            "paper_id": paper.paper_id,
            "paper_version_id": version.paper_version_id,
            "source_pdf_checksum": version.checksum,
            "parsed_artifact_checksum": "irrelevant-legacy-checksum",
            "chunking": {
                "version": "v1", "chunk_size_tokens": 700, "chunk_overlap_tokens": 100,
                "min_chunk_tokens": 80, "tokenizer": "whitespace-v1",
                # no "config_fingerprint" -- the pre-6.1 artifact shape
            },
            "chunks": [],
            "diagnostics": {
                "chunk_count": 0, "min_tokens": 0, "max_tokens": 0, "average_tokens": 0,
                "median_tokens": 0, "small_chunk_count": 0, "oversized_chunk_count": 0,
            },
            "warnings": [],
        }
        legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

        service, chunker = _service(db_session, tmp_path)
        result = service.chunk(paper.paper_id)  # must not raise

        assert chunker.call_count == 1  # a real rechunk happened, safely
        assert result.chunk_reused is False
        assert result.job.status == IngestionStatus.CHUNKED
        assert result.document.chunking.config_fingerprint  # now present


class TestParsedArtifactChecksumChangedInvalidation:
    def test_changed_parsed_checksum_triggers_a_real_rechunk(self, db_session, tmp_path) -> None:
        paper, version, _checksum = _prepare_parsed_paper(db_session, tmp_path, source_id="2401.88840")
        service, chunker = _service(db_session, tmp_path)
        service.chunk(paper.paper_id)
        assert chunker.call_count == 1

        # Simulate the paper having been legitimately reparsed (new
        # parsed.json content) since chunking.
        settings = _settings(tmp_path)
        parsed_storage = ParsedArtifactStorage(settings)
        parser = PyMuPDFParser()
        pdf_storage = PaperStorage(settings)
        pdf_path = pdf_storage.get_path(source="arxiv", source_id=paper.source_id, version="v1")
        document = parser.parse(pdf_path, paper_id=paper.paper_id, paper_version_id=version.paper_version_id)
        # Mutate full_text/sections slightly so the artifact's bytes (and
        # thus its checksum) genuinely differ -- appending to the first
        # section's text is enough to change the serialized JSON.
        mutated_sections = list(document.sections)
        mutated_sections[0] = mutated_sections[0].model_copy(
            update={"text": mutated_sections[0].text + "\n\nAn appended reparse artifact."}
        )
        document = document.model_copy(
            update={"sections": mutated_sections, "source_pdf_checksum": version.checksum}
        )
        parsed_storage.write(document, source="arxiv", source_id=paper.source_id, version="v1")

        result = service.chunk(paper.paper_id)

        assert chunker.call_count == 2  # a genuine rechunk happened
        assert result.chunk_reused is False


class TestReconciliation:
    def test_chunks_json_exists_but_final_db_write_failed_is_reconciled_without_rechunking(
        self, db_session, tmp_path
    ) -> None:
        paper, version, _checksum = _prepare_parsed_paper(db_session, tmp_path, source_id="2401.88850")
        settings = _settings(tmp_path)
        parsed_storage = ParsedArtifactStorage(settings)
        parsed_document = parsed_storage.try_read(
            source="arxiv", source_id=paper.source_id, version="v1"
        )
        parsed_path = parsed_storage.get_path(source="arxiv", source_id=paper.source_id, version="v1")
        parsed_checksum = sha256_file(parsed_path)

        # Chunk directly via the chunker + storage (bypassing the service),
        # simulating "chunks.json was finalized but the service's DB write
        # never happened" -- no ingestion job, no DB metadata recorded.
        chunker = SectionAwareChunker(settings)
        document = chunker.chunk(parsed_document)
        document = document.model_copy(update={"parsed_artifact_checksum": parsed_checksum})
        ChunkArtifactStorage(settings).write(
            document, source="arxiv", source_id=paper.source_id, version="v1"
        )

        counting_chunker = CountingChunker(settings)
        service, _ = _service(db_session, tmp_path, chunker=counting_chunker)
        result = service.chunk(paper.paper_id)

        assert counting_chunker.call_count == 0  # never rechunked -- reconciled from disk
        assert result.chunk_reused is True
        assert result.job.status == IngestionStatus.CHUNKED

        stored_version = PaperRepository(db_session).get_paper_version(version.paper_version_id)
        assert stored_version.chunked_artifact_path is not None
        assert stored_version.chunk_count == len(document.chunks)

    def test_db_says_chunked_but_artifact_missing_triggers_rechunk(self, db_session, tmp_path) -> None:
        paper, _version, _checksum = _prepare_parsed_paper(db_session, tmp_path, source_id="2401.88860")
        service, chunker = _service(db_session, tmp_path)
        first = service.chunk(paper.paper_id)
        assert first.job.status == IngestionStatus.CHUNKED

        settings = _settings(tmp_path)
        ChunkArtifactStorage(settings).delete(source="arxiv", source_id=paper.source_id, version="v1")

        second = service.chunk(paper.paper_id)

        assert chunker.call_count == 2  # missing artifact forced a real rechunk
        assert second.job.status == IngestionStatus.CHUNKED
        assert second.job.ingestion_job_id != first.job.ingestion_job_id  # stale job replaced

    def test_corrupt_chunks_json_triggers_rechunk(self, db_session, tmp_path) -> None:
        paper, _version, _checksum = _prepare_parsed_paper(db_session, tmp_path, source_id="2401.88870")
        service, chunker = _service(db_session, tmp_path)
        first = service.chunk(paper.paper_id)
        assert first.job.status == IngestionStatus.CHUNKED

        settings = _settings(tmp_path)
        chunk_path = ChunkArtifactStorage(settings).get_path(
            source="arxiv", source_id=paper.source_id, version="v1"
        )
        chunk_path.write_text("{ not valid json at all", encoding="utf-8")

        second = service.chunk(paper.paper_id)

        assert chunker.call_count == 2  # corruption forced a real rechunk
        assert second.job.status == IngestionStatus.CHUNKED
        assert second.job.ingestion_job_id != first.job.ingestion_job_id


class TestJobAlreadyPastChunked:
    """Regression coverage (found via Prompt 7's own integration testing):
    once VECTOR_INDEXING/VECTOR_INDEXED became reachable job statuses, a
    paper's active job can legitimately be sitting well past CHUNKED when
    `/chunk` is called again -- reconciliation must handle that without
    crashing or regressing the reported status."""

    def test_rechunking_after_the_job_advanced_past_chunked_does_not_crash(
        self, db_session, tmp_path
    ) -> None:
        paper, version, _checksum = _prepare_parsed_paper(db_session, tmp_path, source_id="2401.88880")
        service, chunker = _service(db_session, tmp_path)
        first = service.chunk(paper.paper_id)
        assert first.job.status == IngestionStatus.CHUNKED

        # Simulate the same job having legitimately advanced further
        # (vector indexing already ran) since this chunk was produced.
        ingestion_repo = IngestionRepository(db_session)
        ingestion_repo.transition_job_status(first.job.ingestion_job_id, IngestionStatus.VECTOR_INDEXING)
        ingestion_repo.transition_job_status(first.job.ingestion_job_id, IngestionStatus.VECTOR_INDEXED)

        # Chunking config is unchanged -- this must reuse the still-valid
        # chunks and must NOT crash with an illegal state transition, and
        # must NOT regress the job's status back down to CHUNKED.
        second = service.chunk(paper.paper_id)

        assert chunker.call_count == 1  # never rechunked -- config didn't change
        assert second.chunk_reused is True
        assert second.job.status == IngestionStatus.VECTOR_INDEXED  # not regressed
        assert second.job.ingestion_job_id == first.job.ingestion_job_id

    def test_rechunking_with_changed_config_after_vector_indexing_starts_a_fresh_job(
        self, db_session, tmp_path
    ) -> None:
        paper, version, _checksum = _prepare_parsed_paper(db_session, tmp_path, source_id="2401.88881")
        service, chunker = _service(db_session, tmp_path)
        first = service.chunk(paper.paper_id)

        ingestion_repo = IngestionRepository(db_session)
        ingestion_repo.transition_job_status(first.job.ingestion_job_id, IngestionStatus.VECTOR_INDEXING)
        ingestion_repo.transition_job_status(first.job.ingestion_job_id, IngestionStatus.VECTOR_INDEXED)

        service2, chunker2 = _service(db_session, tmp_path, CHUNK_SIZE_TOKENS=50)
        second = service2.chunk(paper.paper_id)

        assert chunker2.call_count == 1  # a genuine rechunk happened
        assert second.chunk_reused is False
        assert second.job.status == IngestionStatus.CHUNKED  # correctly reset to CHUNKED
        assert second.job.ingestion_job_id != first.job.ingestion_job_id  # stale job replaced
