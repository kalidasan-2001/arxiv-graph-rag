"""Integration tests for `VectorIndexingService` against real PostgreSQL
and real Qdrant.

No live network, no real embedding model -- the "chunked" precondition is
set up via real `PyMuPDFParser` + `SectionAwareChunker` (matching what
Prompts 5/6 would have already produced), and embeddings come from the
deterministic `FakeEmbeddingProvider` (prompt #55/#56). Requires both a
reachable database and a reachable Qdrant (see `tests/integration/conftest.py`);
skipped automatically otherwise.

Covers the full target flow (prompt #59): chunked paper -> vector-index
request -> CHUNKED -> VECTOR_INDEXING -> Qdrant points -> VECTOR_INDEXED --
plus idempotency, batch-size handling, DB-final-write-failure and
partial-generation reconciliation, embedding-config-changed invalidation,
chunk-artifact-changed invalidation, and dimension-mismatch rejection.
"""

import pytest

from app.core.config import Settings
from app.core.exceptions import ChunkArtifactNotFoundError, VectorCollectionIncompatibleError
from app.domain.enums import IngestionStatus
from app.domain.papers import Paper, PaperVersion
from app.ingestion.checksums import sha256_file
from app.ingestion.chunking.section_chunker import SectionAwareChunker
from app.ingestion.chunking.service import ChunkingService
from app.ingestion.chunking.storage import ChunkArtifactStorage
from app.ingestion.download.storage import PaperStorage
from app.ingestion.parsing.pymupdf_parser import PyMuPDFParser
from app.ingestion.parsing.storage import ParsedArtifactStorage
from app.ingestion.vector_indexing.fingerprint import build_vector_generation_fingerprint
from app.ingestion.vector_indexing.service import VectorIndexingService
from app.storage.postgres.repositories.ingestion import IngestionRepository
from app.storage.postgres.repositories.papers import PaperRepository
from app.storage.qdrant.models import VectorPoint, VectorPointPayload, build_qdrant_point_id
from app.storage.qdrant.qdrant_repository import QdrantVectorRepository
from tests.embeddings.fakes import FakeEmbeddingProvider
from tests.parsing.pdf_fixtures import make_scientific_paper_pdf_bytes


def _settings(storage_root, **overrides) -> Settings:
    return Settings(PAPER_STORAGE_PATH=str(storage_root), **overrides)


def _prepare_chunked_paper(db_session, storage_root, *, source_id: str = "2401.77801"):
    """Discover a paper/version, place a real PDF, parse and chunk it for
    real -- exactly the precondition `VectorIndexingService` expects."""

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
        version.paper_version_id, checksum=pdf_checksum, storage_path=str(final_path),
        file_size_bytes=final_path.stat().st_size, downloaded_at=version.created_at,
    )

    parser = PyMuPDFParser()
    parsed_document = parser.parse(
        final_path, paper_id=paper.paper_id, paper_version_id=version.paper_version_id
    )
    parsed_document = parsed_document.model_copy(update={"source_pdf_checksum": pdf_checksum})
    parsed_storage = ParsedArtifactStorage(settings)
    parsed_path = parsed_storage.write(parsed_document, source="arxiv", source_id=source_id, version="v1")
    version = papers.update_version_parse_result(
        version.paper_version_id, parsed_artifact_path=str(parsed_path), parsed_at=version.created_at,
        parser_name=parsed_document.parser_name, parser_version=parsed_document.parser_version,
        page_count=parsed_document.page_count, section_count=len(parsed_document.sections),
        warning_count=len(parsed_document.warnings),
    )

    chunking_service = ChunkingService(
        SectionAwareChunker(settings), parsed_storage, ChunkArtifactStorage(settings),
        papers, IngestionRepository(db_session),
    )
    chunk_result = chunking_service.chunk(paper.paper_id)
    chunk_path = ChunkArtifactStorage(settings).get_path(source="arxiv", source_id=source_id, version="v1")
    chunk_checksum = sha256_file(chunk_path)

    version = papers.get_paper_version(version.paper_version_id)
    return paper, version, chunk_checksum, len(chunk_result.document.chunks)


def _service(
    db_session, storage_root, qdrant_client, qdrant_collection_name, *,
    provider=None, batch_size: int = 8,
) -> tuple[VectorIndexingService, FakeEmbeddingProvider]:
    settings = _settings(storage_root)
    embedding_provider = provider or FakeEmbeddingProvider(dimension=8, normalize=True)
    vector_repo = QdrantVectorRepository(qdrant_client, qdrant_collection_name)
    service = VectorIndexingService(
        embedding_provider, vector_repo, ChunkArtifactStorage(settings),
        PaperRepository(db_session), IngestionRepository(db_session),
        collection_name=qdrant_collection_name, batch_size=batch_size,
    )
    return service, embedding_provider


class TestFullIndexingFlow:
    def test_chunked_paper_is_indexed_and_marked_vector_indexed(
        self, db_session, tmp_path, qdrant_client, qdrant_collection_name
    ) -> None:
        paper, version, _checksum, expected_chunks = _prepare_chunked_paper(db_session, tmp_path)
        service, provider = _service(db_session, tmp_path, qdrant_client, qdrant_collection_name)

        result = service.index_paper_version(paper.paper_id)

        assert result.job.status == IngestionStatus.VECTOR_INDEXED
        assert result.index_reused is False
        assert result.vector_count == expected_chunks

        stored_version = PaperRepository(db_session).get_paper_version(version.paper_version_id)
        assert stored_version.vector_count == expected_chunks
        assert stored_version.embedding_provider == provider.provider_name
        assert stored_version.vector_generation_fingerprint == result.vector_generation_fingerprint
        assert stored_version.vector_indexed_at is not None

        repo = QdrantVectorRepository(qdrant_client, qdrant_collection_name)
        assert repo.count_for_paper_version(version.paper_version_id) == expected_chunks

    def test_index_without_chunks_raises(self, db_session, tmp_path, qdrant_client, qdrant_collection_name) -> None:
        papers = PaperRepository(db_session)
        paper = papers.upsert_paper(
            Paper.create(source="arxiv", source_id="2401.77802", title="Not Chunked")
        )
        papers.get_or_create_paper_version(PaperVersion.create(paper_id=paper.paper_id, version="v1"))
        service, _provider = _service(db_session, tmp_path, qdrant_client, qdrant_collection_name)

        with pytest.raises(ChunkArtifactNotFoundError):
            service.index_paper_version(paper.paper_id)


class TestBatchSizeHandling:
    def test_chunks_are_embedded_in_configured_batches(
        self, db_session, tmp_path, qdrant_client, qdrant_collection_name
    ) -> None:
        paper, _version, _checksum, expected_chunks = _prepare_chunked_paper(
            db_session, tmp_path, source_id="2401.77803"
        )
        service, provider = _service(
            db_session, tmp_path, qdrant_client, qdrant_collection_name, batch_size=8
        )

        service.index_paper_version(paper.paper_id)

        batch_sizes = [len(batch) for batch in provider.embed_documents_calls]
        assert sum(batch_sizes) == expected_chunks
        assert all(size <= 8 for size in batch_sizes)
        # Never one chunk at a time unless the whole corpus is that small
        # (prompt #22) -- with a real multi-section test fixture and
        # batch_size=8, more than one chunk per call is expected.
        assert any(size > 1 for size in batch_sizes)


class TestIdempotency:
    def test_second_index_does_not_invoke_the_provider_again(
        self, db_session, tmp_path, qdrant_client, qdrant_collection_name
    ) -> None:
        paper, _version, _checksum, _count = _prepare_chunked_paper(
            db_session, tmp_path, source_id="2401.77810"
        )
        service, provider = _service(db_session, tmp_path, qdrant_client, qdrant_collection_name)

        first = service.index_paper_version(paper.paper_id)
        calls_after_first = provider.call_count
        second = service.index_paper_version(paper.paper_id)

        assert provider.call_count == calls_after_first  # the critical assertion
        assert second.index_reused is True
        assert second.job.status == IngestionStatus.VECTOR_INDEXED
        assert second.job.ingestion_job_id == first.job.ingestion_job_id
        assert second.vector_count == first.vector_count


class TestReconciliation:
    def test_qdrant_complete_but_final_db_write_failed_is_reconciled_without_reembedding(
        self, db_session, tmp_path, qdrant_client, qdrant_collection_name
    ) -> None:
        paper, version, chunk_checksum, _count = _prepare_chunked_paper(
            db_session, tmp_path, source_id="2401.77820"
        )
        settings = _settings(tmp_path)
        provider = FakeEmbeddingProvider(dimension=8, normalize=True)
        vector_repo = QdrantVectorRepository(qdrant_client, qdrant_collection_name)

        # Index directly via the embedding provider + Qdrant repository,
        # bypassing `VectorIndexingService` entirely -- simulates "Qdrant
        # was fully populated but the service's final PostgreSQL write
        # never happened": no ingestion job, no vector metadata recorded.
        chunk_document = ChunkArtifactStorage(settings).try_read(
            source="arxiv", source_id=paper.source_id, version="v1"
        )
        embedding_config_fingerprint = provider.config_fingerprint
        generation_fingerprint = build_vector_generation_fingerprint(
            chunk_artifact_checksum=chunk_checksum,
            embedding_config_fingerprint=embedding_config_fingerprint,
        )
        vector_repo.ensure_collection(dimension=provider.dimension, distance="cosine")
        vectors = provider.embed_documents([chunk.text for chunk in chunk_document.chunks])
        points = [
            VectorPoint(
                point_id=build_qdrant_point_id(chunk.chunk_id),
                vector=vector,
                payload=VectorPointPayload(
                    chunk_id=chunk.chunk_id, paper_id=chunk.paper_id,
                    paper_version_id=chunk.paper_version_id, section_id=chunk.section_id,
                    section_type=chunk.section_type.value,
                    section_title=chunk.metadata.get("section_title"), chunk_index=chunk.chunk_index,
                    page_start=chunk.page_start, page_end=chunk.page_end, source=paper.source,
                    source_id=paper.source_id, published_year=None, categories=list(paper.categories),
                    chunking_version=chunk_document.chunking.version,
                    chunk_config_fingerprint=chunk_document.chunking.config_fingerprint,
                    embedding_provider=provider.provider_name, embedding_model=provider.model_name,
                    embedding_config_fingerprint=embedding_config_fingerprint,
                    vector_generation_fingerprint=generation_fingerprint, text=chunk.text,
                ),
            )
            for chunk, vector in zip(chunk_document.chunks, vectors, strict=True)
        ]
        vector_repo.upsert_chunks(points)
        assert version.vector_generation_fingerprint is None  # nothing recorded in Postgres yet

        counting_provider = FakeEmbeddingProvider(dimension=8, normalize=True)
        service = VectorIndexingService(
            counting_provider, vector_repo, ChunkArtifactStorage(settings),
            PaperRepository(db_session), IngestionRepository(db_session),
            collection_name=qdrant_collection_name, batch_size=8,
        )
        result = service.index_paper_version(paper.paper_id)

        assert counting_provider.call_count == 0  # never re-embedded -- reconciled from Qdrant
        assert result.index_reused is True
        assert result.job.status == IngestionStatus.VECTOR_INDEXED

        stored_version = PaperRepository(db_session).get_paper_version(version.paper_version_id)
        assert stored_version.vector_count == len(points)
        assert stored_version.vector_generation_fingerprint == generation_fingerprint

    def test_db_says_vector_indexed_but_qdrant_missing_triggers_reindex(
        self, db_session, tmp_path, qdrant_client, qdrant_collection_name
    ) -> None:
        paper, version, _checksum, expected_chunks = _prepare_chunked_paper(
            db_session, tmp_path, source_id="2401.77830"
        )
        service, provider = _service(db_session, tmp_path, qdrant_client, qdrant_collection_name)
        first = service.index_paper_version(paper.paper_id)
        assert first.job.status == IngestionStatus.VECTOR_INDEXED

        vector_repo = QdrantVectorRepository(qdrant_client, qdrant_collection_name)
        deleted = vector_repo.delete_paper_version(version.paper_version_id)
        assert deleted == expected_chunks

        second = service.index_paper_version(paper.paper_id)

        assert provider.call_count > 0  # a real reindex happened
        assert second.job.status == IngestionStatus.VECTOR_INDEXED
        assert second.job.ingestion_job_id != first.job.ingestion_job_id  # stale job replaced
        assert vector_repo.count_for_paper_version(version.paper_version_id) == expected_chunks

    def test_partial_current_generation_triggers_repair(
        self, db_session, tmp_path, qdrant_client, qdrant_collection_name
    ) -> None:
        paper, version, _checksum, expected_chunks = _prepare_chunked_paper(
            db_session, tmp_path, source_id="2401.77840"
        )
        assert expected_chunks > 1, "fixture must produce more than one chunk for this test to mean anything"
        service, provider = _service(db_session, tmp_path, qdrant_client, qdrant_collection_name)
        first = service.index_paper_version(paper.paper_id)

        # Simulate a partial Qdrant failure: manually remove every point
        # for this paper version, without touching PostgreSQL, then
        # re-upsert only one of them directly -- leaving the collection in
        # a genuine 0 < current < expected state without going through the
        # service (which would otherwise "fix" it for us).
        vector_repo = QdrantVectorRepository(qdrant_client, qdrant_collection_name)
        vector_repo.delete_paper_version(version.paper_version_id)
        assert vector_repo.count_for_paper_version(version.paper_version_id) == 0

        chunk_document = ChunkArtifactStorage(_settings(tmp_path)).try_read(
            source="arxiv", source_id=paper.source_id, version="v1"
        )
        one_chunk = chunk_document.chunks[0]
        one_point = VectorPoint(
            point_id=build_qdrant_point_id(one_chunk.chunk_id),
            vector=provider.embed_documents([one_chunk.text])[0],
            payload=VectorPointPayload(
                chunk_id=one_chunk.chunk_id,
                paper_id=one_chunk.paper_id,
                paper_version_id=one_chunk.paper_version_id,
                section_id=one_chunk.section_id,
                section_type=one_chunk.section_type.value,
                section_title=one_chunk.metadata.get("section_title"),
                chunk_index=one_chunk.chunk_index,
                page_start=one_chunk.page_start,
                page_end=one_chunk.page_end,
                source=paper.source,
                source_id=paper.source_id,
                published_year=None,
                categories=list(paper.categories),
                chunking_version=chunk_document.chunking.version,
                chunk_config_fingerprint=chunk_document.chunking.config_fingerprint,
                embedding_provider=provider.provider_name,
                embedding_model=provider.model_name,
                embedding_config_fingerprint=first.embedding_config_fingerprint,
                vector_generation_fingerprint=first.vector_generation_fingerprint,
                text=one_chunk.text,
            ),
        )
        vector_repo.upsert_chunks([one_point])
        assert vector_repo.count_for_paper_version(version.paper_version_id) == 1

        second = service.index_paper_version(paper.paper_id)

        assert second.job.status == IngestionStatus.VECTOR_INDEXED
        assert second.vector_count == expected_chunks
        assert vector_repo.count_for_paper_version(version.paper_version_id) == expected_chunks
        assert second.job.ingestion_job_id != first.job.ingestion_job_id


class TestEmbeddingConfigChangedInvalidation:
    def test_different_embedding_model_triggers_a_real_reindex(
        self, db_session, tmp_path, qdrant_client, qdrant_collection_name
    ) -> None:
        paper, version, _checksum, expected_chunks = _prepare_chunked_paper(
            db_session, tmp_path, source_id="2401.77850"
        )
        provider_a = FakeEmbeddingProvider(dimension=8, model_name="model-a", normalize=True)
        service_a, _ = _service(
            db_session, tmp_path, qdrant_client, qdrant_collection_name, provider=provider_a
        )
        first = service_a.index_paper_version(paper.paper_id)
        assert provider_a.call_count > 0

        provider_b = FakeEmbeddingProvider(dimension=8, model_name="model-b", normalize=True)
        service_b, _ = _service(
            db_session, tmp_path, qdrant_client, qdrant_collection_name, provider=provider_b
        )
        second = service_b.index_paper_version(paper.paper_id)

        assert provider_b.call_count > 0  # a genuine re-embed happened
        assert second.index_reused is False
        assert second.embedding_config_fingerprint != first.embedding_config_fingerprint

        vector_repo = QdrantVectorRepository(qdrant_client, qdrant_collection_name)
        # Old generation's points must be gone -- they must not remain as
        # current-generation search results (prompt #63).
        assert (
            vector_repo.count_for_paper_version(
                version.paper_version_id, generation_fingerprint=first.vector_generation_fingerprint
            )
            == 0
        )
        assert (
            vector_repo.count_for_paper_version(
                version.paper_version_id, generation_fingerprint=second.vector_generation_fingerprint
            )
            == expected_chunks
        )


class TestChunkArtifactChangedInvalidation:
    def test_rechunking_triggers_a_real_reindex(
        self, db_session, tmp_path, qdrant_client, qdrant_collection_name
    ) -> None:
        paper, version, _checksum, _count = _prepare_chunked_paper(
            db_session, tmp_path, source_id="2401.77860"
        )
        service, provider = _service(db_session, tmp_path, qdrant_client, qdrant_collection_name)
        first = service.index_paper_version(paper.paper_id)
        assert provider.call_count > 0

        # Simulate the paper having been legitimately rechunked (a
        # different CHUNK_SIZE_TOKENS produces different chunks.json content).
        rechunk_settings = _settings(tmp_path, CHUNK_SIZE_TOKENS=50)
        rechunk_service = ChunkingService(
            SectionAwareChunker(rechunk_settings), ParsedArtifactStorage(rechunk_settings),
            ChunkArtifactStorage(rechunk_settings), PaperRepository(db_session),
            IngestionRepository(db_session),
        )
        rechunk_result = rechunk_service.chunk(paper.paper_id)
        assert rechunk_result.chunk_reused is False

        second = service.index_paper_version(paper.paper_id)

        assert provider.call_count > 0
        assert second.index_reused is False
        assert second.vector_generation_fingerprint != first.vector_generation_fingerprint

        vector_repo = QdrantVectorRepository(qdrant_client, qdrant_collection_name)
        assert (
            vector_repo.count_for_paper_version(
                version.paper_version_id, generation_fingerprint=first.vector_generation_fingerprint
            )
            == 0
        )


class TestDimensionMismatch:
    def test_incompatible_existing_collection_raises_and_does_not_destroy_it(
        self, db_session, tmp_path, qdrant_client, qdrant_collection_name
    ) -> None:
        paper, _version, _checksum, _count = _prepare_chunked_paper(
            db_session, tmp_path, source_id="2401.77870"
        )
        provider_384 = FakeEmbeddingProvider(dimension=384, normalize=True)
        service_a, _ = _service(
            db_session, tmp_path, qdrant_client, qdrant_collection_name, provider=provider_384
        )
        service_a.index_paper_version(paper.paper_id)

        provider_768 = FakeEmbeddingProvider(dimension=768, normalize=True)
        service_b, _ = _service(
            db_session, tmp_path, qdrant_client, qdrant_collection_name, provider=provider_768,
        )

        with pytest.raises(VectorCollectionIncompatibleError):
            service_b.index_paper_version(paper.paper_id)

        info = qdrant_client.get_collection(qdrant_collection_name)
        assert info.config.params.vectors.size == 384  # never silently destroyed/recreated
