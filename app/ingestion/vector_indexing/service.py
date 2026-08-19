"""Vector indexing service: orchestrates embedding, Qdrant upsert, and
idempotency/reconciliation for the vector-indexing stage.

Target flow (prompt #20)::

    CHUNKED -> validate chunk artifact -> VECTOR_INDEXING -> EmbeddingProvider
    -> batch embeddings -> Qdrant upsert -> verify -> Postgres metadata
    -> VECTOR_INDEXED

Never continues into graph extraction.

Reconciliation mirrors Prompt 6/6.1's filesystem-first design, but the
external source of truth here is Qdrant, not a local file: the *current*
count of points stamped with the *current* `vector_generation_fingerprint`
(prompt #31) is what "has this already been indexed" actually means, never
whatever PostgreSQL currently records (prompt #29).

Stale-generation policy (prompt #32): upsert the complete new generation
first, verify it, and only then delete points from the old generation --
the "safer alternative" the prompt prefers over delete-then-upsert, since a
paper's vectors are never zero mid-reindex if something fails partway.
"""

import logging
import time
from datetime import datetime, timezone

from pydantic import BaseModel

from app.core.exceptions import ChunkArtifactNotFoundError, VectorIndexingError
from app.domain.enums import IngestionStatus, ProcessingStage, StepStatus
from app.domain.ingestion import IngestionJob, IngestionStepState
from app.domain.papers import Paper, PaperChunk, PaperVersion
from app.embeddings.provider import EmbeddingProvider
from app.ingestion.chunking.models import ChunkedPaperDocument
from app.ingestion.chunking.storage import ChunkArtifactStorage
from app.ingestion.checksums import sha256_file
from app.ingestion.paper_resolution import resolve_paper, resolve_version
from app.ingestion.vector_indexing.fingerprint import build_vector_generation_fingerprint
from app.storage.postgres.repositories.ingestion import IngestionRepository
from app.storage.postgres.repositories.papers import PaperRepository
from app.storage.qdrant.models import VectorPoint, VectorPointPayload, build_qdrant_point_id
from app.storage.qdrant.repository import VectorRepository

logger = logging.getLogger(__name__)

_DISTANCE_METRIC = "cosine"  # see QdrantVectorRepository -- V1's only supported metric

# Every status a job can hold *at or after* VECTOR_INDEXED (prompt 8 will
# add GRAPH_INDEXING/GRAPH_INDEXED). Written from the start using the same
# "at or past this stage" set, not a bare `== VECTOR_INDEXED` check -- an
# equivalent bug was found and fixed in `ChunkingService` during this
# prompt's own integration testing (re-chunking a paper already past
# CHUNKED crashed with an illegal transition), so `VectorIndexingService`
# is written to not repeat it once graph indexing exists.
_STATUSES_AT_OR_PAST_VECTOR_INDEXED = frozenset(
    {
        IngestionStatus.VECTOR_INDEXED,
        IngestionStatus.GRAPH_INDEXING,
        IngestionStatus.GRAPH_INDEXED,
    }
)


class VectorIndexResult(BaseModel):
    """Outcome of `VectorIndexingService.index_paper_version()`."""

    job: IngestionJob
    vector_count: int
    embedding_provider: str
    embedding_model: str
    embedding_config_fingerprint: str
    vector_generation_fingerprint: str
    index_reused: bool = False


class VectorIndexingService:
    """Explicit vector indexing: the only thing FastAPI routes should call
    for `POST /api/v1/papers/{paper_id}/vector-index`."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_repository: VectorRepository,
        chunk_storage: ChunkArtifactStorage,
        paper_repository: PaperRepository,
        ingestion_repository: IngestionRepository,
        *,
        collection_name: str,
        batch_size: int,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_repo = vector_repository
        self._chunk_storage = chunk_storage
        self._paper_repo = paper_repository
        self._ingestion_repo = ingestion_repository
        self._collection_name = collection_name
        self._batch_size = batch_size

    def index_paper_version(
        self, paper_id: str, paper_version_id: str | None = None
    ) -> VectorIndexResult:
        paper = resolve_paper(self._paper_repo, paper_id)
        version = resolve_version(self._paper_repo, paper, paper_version_id)
        chunk_document, chunk_checksum = self._validate_ready_for_indexing(paper, version)

        # Computing `.config_fingerprint` may trigger the one-time lazy
        # model load (prompt #43) -- unavoidable: dimension genuinely isn't
        # knowable without it, and every path below needs the fingerprint.
        embedding_config_fingerprint = self._embedding_provider.config_fingerprint
        generation_fingerprint = build_vector_generation_fingerprint(
            chunk_artifact_checksum=chunk_checksum,
            embedding_config_fingerprint=embedding_config_fingerprint,
        )

        self._vector_repo.ensure_collection(
            dimension=self._embedding_provider.dimension, distance=_DISTANCE_METRIC
        )

        job = self._ingestion_repo.create_ingestion_job(
            paper_id=paper.paper_id, paper_version_id=version.paper_version_id
        )

        result, job = self._reconcile_existing_index(
            paper,
            version,
            job,
            chunk_document=chunk_document,
            generation_fingerprint=generation_fingerprint,
            embedding_config_fingerprint=embedding_config_fingerprint,
        )
        if result is not None:
            return result

        return self._index_and_finalize(
            paper,
            version,
            job,
            chunk_document=chunk_document,
            generation_fingerprint=generation_fingerprint,
            embedding_config_fingerprint=embedding_config_fingerprint,
        )

    # --- Validation -----------------------------------------------------

    def _validate_ready_for_indexing(
        self, paper: Paper, version: PaperVersion
    ) -> tuple[ChunkedPaperDocument, str]:
        """Prompt #51: chunk identity/consistency checks before any
        embedding happens. `chunks.json` failing schema validation
        (including a legacy artifact missing `chunk_config_fingerprint`,
        required since Prompt 6.1) already surfaces as `try_read()`
        returning `None` -- that case requires re-chunking, not guessing,
        exactly like `ChunkArtifactNotFoundError` already communicates."""

        document = self._chunk_storage.try_read(
            source=paper.source, source_id=paper.source_id, version=version.version
        )
        if document is None:
            raise ChunkArtifactNotFoundError(
                f"no valid chunk artifact for {version.paper_version_id}; chunk it first"
            )
        if document.paper_id != paper.paper_id or document.paper_version_id != version.paper_version_id:
            raise VectorIndexingError(
                f"chunk artifact identity mismatch for {version.paper_version_id}"
            )
        if not document.chunks:
            raise VectorIndexingError(
                f"chunk artifact for {version.paper_version_id} has zero chunks; nothing to index"
            )
        if version.chunk_count is not None and version.chunk_count != len(document.chunks):
            raise VectorIndexingError(
                f"chunk_count mismatch for {version.paper_version_id}: PostgreSQL records "
                f"{version.chunk_count}, chunks.json has {len(document.chunks)}"
            )
        _validate_chunk_identity(document.chunks)

        chunk_path = self._chunk_storage.get_path(
            source=paper.source, source_id=paper.source_id, version=version.version
        )
        return document, sha256_file(chunk_path)

    # --- Reconciliation ---------------------------------------------------

    def _reconcile_existing_index(
        self,
        paper: Paper,
        version: PaperVersion,
        job: IngestionJob,
        *,
        chunk_document: ChunkedPaperDocument,
        generation_fingerprint: str,
        embedding_config_fingerprint: str,
    ) -> tuple[VectorIndexResult | None, IngestionJob]:
        """Filesystem-first reconciliation, applied to Qdrant instead of a
        local file: `count_for_paper_version` re-verified against the
        *current* generation fingerprint is the source of truth for "has
        this already been indexed", never whatever PostgreSQL believes."""

        expected_count = len(chunk_document.chunks)
        current_count = self._vector_repo.count_for_paper_version(
            version.paper_version_id, generation_fingerprint=generation_fingerprint
        )

        if current_count != expected_count:
            if current_count == 0:
                reason = "no current-generation vectors found in Qdrant"
            else:
                reason = (
                    f"partial current-generation vectors in Qdrant "
                    f"({current_count}/{expected_count})"
                )
            logger.info(
                "existing vector index is stale/incomplete, reindexing paper_id=%s "
                "paper_version_id=%s reason=%r status=stale",
                paper.paper_id,
                version.paper_version_id,
                reason,
            )
            return None, self._recreate_job_if_stale(paper, version, job, reason=reason)

        # Valid, current-generation vectors are already fully present --
        # reconcile PostgreSQL to match (covers "Qdrant complete but the
        # final DB write failed") without re-embedding.
        was_already_recorded = version.vector_generation_fingerprint == generation_fingerprint
        self._paper_repo.update_version_vector_result(
            version.paper_version_id,
            vector_indexed_at=version.vector_indexed_at or datetime.now(timezone.utc),
            vector_count=expected_count,
            embedding_provider=self._embedding_provider.provider_name,
            embedding_model=self._embedding_provider.model_name,
            embedding_config_fingerprint=embedding_config_fingerprint,
            vector_generation_fingerprint=generation_fingerprint,
            qdrant_collection=self._collection_name,
        )
        job = self._advance_job_to_vector_indexed(job)

        if not was_already_recorded:
            now = datetime.now(timezone.utc)
            self._record_step(
                job.ingestion_job_id,
                self._next_attempt_number(job.ingestion_job_id),
                StepStatus.COMPLETED,
                started_at=now,
                completed_at=now,
                metadata={"vector_count": expected_count, "reconciled": True},
            )

        logger.info(
            "reused existing valid vector index paper_id=%s paper_version_id=%s "
            "ingestion_job_id=%s status=ok",
            paper.paper_id,
            version.paper_version_id,
            job.ingestion_job_id,
        )
        return (
            VectorIndexResult(
                job=job,
                vector_count=expected_count,
                embedding_provider=self._embedding_provider.provider_name,
                embedding_model=self._embedding_provider.model_name,
                embedding_config_fingerprint=embedding_config_fingerprint,
                vector_generation_fingerprint=generation_fingerprint,
                index_reused=True,
            ),
            job,
        )

    def _recreate_job_if_stale(
        self, paper: Paper, version: PaperVersion, job: IngestionJob, *, reason: str
    ) -> IngestionJob:
        """If `job` had already reached VECTOR_INDEXED (or any later stage)
        but Qdrant no longer backs that claim, mark it failed and start a
        fresh job -- then fast-forward it to CHUNKED, since the chunk
        source was already re-verified valid; only indexing needs redoing."""

        if job.status not in _STATUSES_AT_OR_PAST_VECTOR_INDEXED:
            return job

        self._ingestion_repo.mark_job_failed(
            job.ingestion_job_id, failed_stage=job.status, failure_reason=reason
        )
        logger.warning(
            "marked stale vector-index job failed and starting a new one paper_id=%s "
            "paper_version_id=%s ingestion_job_id=%s reason=%r status=stale",
            paper.paper_id,
            version.paper_version_id,
            job.ingestion_job_id,
            reason,
        )
        new_job = self._ingestion_repo.create_ingestion_job(
            paper_id=paper.paper_id, paper_version_id=version.paper_version_id
        )
        return self._ensure_job_at_chunked(new_job)

    def _ensure_job_at_chunked(self, job: IngestionJob) -> IngestionJob:
        if job.status == IngestionStatus.DISCOVERED:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.DOWNLOADING
            )
        if job.status == IngestionStatus.DOWNLOADING:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.DOWNLOADED
            )
        if job.status == IngestionStatus.DOWNLOADED:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.PARSING
            )
        if job.status == IngestionStatus.PARSING:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.PARSED
            )
        if job.status == IngestionStatus.PARSED:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.CHUNKING
            )
        if job.status == IngestionStatus.CHUNKING:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.CHUNKED
            )
        return job

    def _advance_job_to_vector_indexed(self, job: IngestionJob) -> IngestionJob:
        """No-op if `job` is already at VECTOR_INDEXED or a later stage --
        reusing a still-valid vector generation on a paper that has
        progressed further (e.g. into graph indexing) must never regress
        its reported status (mirrors `ChunkingService._advance_job_to_chunked`)."""

        if job.status in _STATUSES_AT_OR_PAST_VECTOR_INDEXED:
            return job

        job = self._ensure_job_at_chunked(job)
        if job.status == IngestionStatus.CHUNKED:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.VECTOR_INDEXING
            )
        if job.status == IngestionStatus.VECTOR_INDEXING:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.VECTOR_INDEXED
            )
        return job

    # --- Indexing ---------------------------------------------------------

    def _index_and_finalize(
        self,
        paper: Paper,
        version: PaperVersion,
        job: IngestionJob,
        *,
        chunk_document: ChunkedPaperDocument,
        generation_fingerprint: str,
        embedding_config_fingerprint: str,
    ) -> VectorIndexResult:
        started_monotonic = time.monotonic()
        started_at = datetime.now(timezone.utc)
        attempt = self._next_attempt_number(job.ingestion_job_id)

        job = self._ensure_job_at_chunked(job)
        if job.status == IngestionStatus.CHUNKED:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.VECTOR_INDEXING
            )

        try:
            vectors = self._embed_all(chunk_document.chunks)
            points = self._build_points(
                paper,
                version,
                chunk_document,
                vectors,
                generation_fingerprint=generation_fingerprint,
                embedding_config_fingerprint=embedding_config_fingerprint,
            )
            self._vector_repo.upsert_chunks(points)

            # Verify before trusting the write (prompt #24/#29): a partial
            # Qdrant failure must never be reported as success.
            actual_count = self._vector_repo.count_for_paper_version(
                version.paper_version_id, generation_fingerprint=generation_fingerprint
            )
            if actual_count != len(points):
                raise VectorIndexingError(
                    f"post-upsert verification failed for {version.paper_version_id}: "
                    f"expected {len(points)} current-generation points, Qdrant reports {actual_count}"
                )

            # Only now -- generation verified complete -- remove any
            # previous generation's points for this paper version (prompt
            # #32's safer ordering: upsert, verify, then delete stale).
            deleted = self._vector_repo.delete_paper_version(
                version.paper_version_id, exclude_generation_fingerprint=generation_fingerprint
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - started_monotonic) * 1000)
            self._record_step(
                job.ingestion_job_id,
                attempt,
                StepStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                error_code=type(exc).__name__,
                error_message=str(exc)[:500],
                metadata={"duration_ms": duration_ms},
            )
            self._ingestion_repo.mark_job_failed(
                job.ingestion_job_id,
                failed_stage=IngestionStatus.VECTOR_INDEXING,
                failure_reason=str(exc)[:2000],
            )
            logger.warning(
                "vector indexing failed paper_id=%s paper_version_id=%s ingestion_job_id=%s "
                "attempt=%d duration_ms=%d status=error error=%s",
                paper.paper_id,
                version.paper_version_id,
                job.ingestion_job_id,
                attempt,
                duration_ms,
                exc,
            )
            raise

        self._paper_repo.update_version_vector_result(
            version.paper_version_id,
            vector_indexed_at=datetime.now(timezone.utc),
            vector_count=len(points),
            embedding_provider=self._embedding_provider.provider_name,
            embedding_model=self._embedding_provider.model_name,
            embedding_config_fingerprint=embedding_config_fingerprint,
            vector_generation_fingerprint=generation_fingerprint,
            qdrant_collection=self._collection_name,
        )
        job = self._ingestion_repo.transition_job_status(
            job.ingestion_job_id, IngestionStatus.VECTOR_INDEXED
        )

        duration_ms = int((time.monotonic() - started_monotonic) * 1000)
        self._record_step(
            job.ingestion_job_id,
            attempt,
            StepStatus.COMPLETED,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            metadata={
                "vector_count": len(points),
                "stale_points_deleted": deleted,
                "duration_ms": duration_ms,
            },
        )
        logger.info(
            "vector indexing succeeded paper_id=%s paper_version_id=%s ingestion_job_id=%s "
            "embedding_provider=%s embedding_model=%s vector_count=%d "
            "stale_points_deleted=%d duration_ms=%d index_reused=False status=ok",
            paper.paper_id,
            version.paper_version_id,
            job.ingestion_job_id,
            self._embedding_provider.provider_name,
            self._embedding_provider.model_name,
            len(points),
            deleted,
            duration_ms,
        )
        return VectorIndexResult(
            job=job,
            vector_count=len(points),
            embedding_provider=self._embedding_provider.provider_name,
            embedding_model=self._embedding_provider.model_name,
            embedding_config_fingerprint=embedding_config_fingerprint,
            vector_generation_fingerprint=generation_fingerprint,
            index_reused=False,
        )

    def _embed_all(self, chunks: list[PaperChunk]) -> list[list[float]]:
        """Batches embedding calls at `EMBEDDING_BATCH_SIZE` (prompt #22) --
        never one chunk at a time. Every batch's output is re-verified for
        count (prompt #23) even though the provider already validates its
        own output; this is the orchestration-level invariant check."""

        vectors: list[list[float]] = []
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            texts = [chunk.text for chunk in batch]
            batch_vectors = self._embedding_provider.embed_documents(texts)
            if len(batch_vectors) != len(texts):
                raise VectorIndexingError(
                    f"embedding provider returned {len(batch_vectors)} vectors for "
                    f"{len(texts)} texts in one batch"
                )
            vectors.extend(batch_vectors)
        return vectors

    def _build_points(
        self,
        paper: Paper,
        version: PaperVersion,
        chunk_document: ChunkedPaperDocument,
        vectors: list[list[float]],
        *,
        generation_fingerprint: str,
        embedding_config_fingerprint: str,
    ) -> list[VectorPoint]:
        published_year = paper.published_at.year if paper.published_at else None
        points: list[VectorPoint] = []
        for chunk, vector in zip(chunk_document.chunks, vectors, strict=True):
            payload = VectorPointPayload(
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                paper_version_id=chunk.paper_version_id,
                section_id=chunk.section_id,
                section_type=chunk.section_type.value,
                section_title=chunk.metadata.get("section_title"),
                chunk_index=chunk.chunk_index,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                source=paper.source,
                source_id=paper.source_id,
                published_year=published_year,
                categories=list(paper.categories),
                chunking_version=chunk_document.chunking.version,
                chunk_config_fingerprint=chunk_document.chunking.config_fingerprint,
                embedding_provider=self._embedding_provider.provider_name,
                embedding_model=self._embedding_provider.model_name,
                embedding_config_fingerprint=embedding_config_fingerprint,
                vector_generation_fingerprint=generation_fingerprint,
                text=chunk.text,
            )
            points.append(
                VectorPoint(
                    point_id=build_qdrant_point_id(chunk.chunk_id), vector=vector, payload=payload
                )
            )
        return points

    # --- Steps -------------------------------------------------------------

    def _next_attempt_number(self, ingestion_job_id: str) -> int:
        existing = [
            step
            for step in self._ingestion_repo.list_steps(ingestion_job_id)
            if step.stage == ProcessingStage.VECTOR_INDEX
        ]
        return max((step.attempt for step in existing), default=0) + 1

    def _record_step(
        self,
        ingestion_job_id: str,
        attempt: int,
        status: StepStatus,
        *,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._ingestion_repo.record_step(
            IngestionStepState(
                ingestion_job_id=ingestion_job_id,
                stage=ProcessingStage.VECTOR_INDEX,
                status=status,
                attempt=attempt,
                started_at=started_at,
                completed_at=completed_at,
                error_code=error_code,
                error_message=error_message,
                metadata=metadata or {},
            )
        )


def _validate_chunk_identity(chunks: list[PaperChunk]) -> None:
    """Prompt #51: duplicate-id and empty-text checks before any embedding
    work happens -- a chunker bug slipping through would otherwise waste an
    embedding call before failing."""

    seen_ids: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen_ids:
            raise VectorIndexingError(f"duplicate chunk_id in chunk artifact: {chunk.chunk_id}")
        seen_ids.add(chunk.chunk_id)
        if not chunk.text.strip():
            raise VectorIndexingError(f"empty chunk text for {chunk.chunk_id}")
