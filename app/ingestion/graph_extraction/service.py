"""Scientific knowledge extraction service: orchestrates deterministic
paper/author/citation facts, LLM-based semantic extraction, validation,
and idempotency/reconciliation for the graph-extraction stage.

Target flow (prompt #3/#33)::

    VECTOR_INDEXED -> validate chunk artifact -> GRAPH_INDEXING
    -> deterministic Paper/Author/CITES facts
    -> LLM semantic extraction (per chunk, selected sections only)
    -> ontology validation -> deduplication -> provenance attachment
    -> atomic graph_extraction.json -> persist metadata
    -> (job stays at GRAPH_INDEXING -- GRAPH_INDEXED is never claimed; see
       module docstring in app/domain/ingestion.py and the "State Machine
       Placement" note in docs/ARCHITECTURE.md)

Never writes to Neo4j.

Reconciliation mirrors Prompt 6's filesystem-first design (the extraction
artifact, like chunks.json, is a local file): `graph_extraction.json` on
disk, re-verified against the *current* `graph_extraction_generation_fingerprint`
(chunk checksum + extraction config), is the source of truth for "has this
already been extracted", never whatever PostgreSQL currently records.

Partial-failure policy (prompt #39): the artifact is only ever written
after every selected chunk has been successfully processed. Any chunk's
LLM call failing after retries aborts the whole extraction -- there is no
"partially trusted" `graph_extraction.json`.
"""

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone

from pydantic import BaseModel

from app.core.exceptions import ChunkArtifactNotFoundError, GraphExtractionError
from app.domain.enums import IngestionStatus, ProcessingStage, SectionType, StepStatus
from app.domain.enums import EntityType, RelationshipType
from app.domain.ids import normalize_identity_key, normalize_whitespace
from app.domain.ingestion import IngestionJob, IngestionStepState
from app.domain.knowledge import ScientificEntity, ScientificRelationship
from app.domain.papers import Paper, PaperChunk, PaperVersion
from app.ingestion.checksums import sha256_file
from app.ingestion.chunking.models import ChunkedPaperDocument
from app.ingestion.chunking.storage import ChunkArtifactStorage
from app.ingestion.graph_extraction.citations import resolve_citations
from app.ingestion.graph_extraction.fingerprint import (
    build_extraction_config_fingerprint,
    build_graph_extraction_generation_fingerprint,
)
from app.ingestion.graph_extraction.models import (
    ExtractedEntityCandidate,
    ExtractedRelationshipCandidate,
    ExtractionConfig,
    GraphExtractionArtifact,
    GraphExtractionWarning,
    GraphExtractionWarningCode,
    RawExtractionResponse,
    UnresolvedCitation,
)
from app.ingestion.graph_extraction.ontology import (
    validate_entity_candidate,
    validate_relationship_candidate,
)
from app.ingestion.graph_extraction.prompt import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    build_system_prompt,
    build_user_prompt,
)
from app.ingestion.graph_extraction.storage import GraphExtractionArtifactStorage
from app.ingestion.paper_resolution import resolve_paper, resolve_version
from app.llm.provider import LLMProvider
from app.storage.postgres.repositories.ingestion import IngestionRepository
from app.storage.postgres.repositories.papers import PaperRepository

logger = logging.getLogger(__name__)

# Sections worth spending an LLM call on for semantic extraction (prompt
# #34). RELATED_WORK is deliberately *excluded* -- not merely relying on
# the "used_by_this_paper" vs "mentioned_only" prompt instruction, but
# structurally: a method/dataset discussed only in Related Work never even
# reaches the model, so it cannot become a false-positive USES_METHOD/
# EVALUATED_ON (prompt #13's "this is critical"). LIMITATIONS is likewise
# excluded (rarely introduces genuine new usage claims). REFERENCES is
# handled separately, deterministically, for citation resolution only.
_SEMANTIC_EXTRACTION_SECTIONS = frozenset(
    {
        SectionType.ABSTRACT,
        SectionType.INTRODUCTION,
        SectionType.METHODOLOGY,
        SectionType.EXPERIMENTS,
        SectionType.RESULTS,
        SectionType.DISCUSSION,
        SectionType.CONCLUSION,
        SectionType.OTHER,
    }
)

# Every status a job can hold *at or after* GRAPH_INDEXING -- mirrors
# `ChunkingService`'s `_STATUSES_AT_OR_PAST_CHUNKED` (a real bug found
# during Prompt 7's integration testing) and `VectorIndexingService`'s
# equivalent, applied here proactively before Prompt 9 makes GRAPH_INDEXED
# reachable.
_STATUSES_AT_OR_PAST_GRAPH_INDEXING = frozenset(
    {IngestionStatus.GRAPH_INDEXING, IngestionStatus.GRAPH_INDEXED}
)


class GraphExtractionResult(BaseModel):
    """Outcome of `ScientificKnowledgeExtractionService.extract()`."""

    job: IngestionJob
    entity_count: int
    relationship_count: int
    unresolved_citation_count: int
    extraction_version: str
    extraction_config_fingerprint: str
    graph_extraction_generation_fingerprint: str
    extraction_reused: bool = False


class ScientificKnowledgeExtractionService:
    """Explicit graph extraction: the only thing FastAPI routes should
    call for `POST /api/v1/papers/{paper_id}/extract-graph`."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        chunk_storage: ChunkArtifactStorage,
        extraction_storage: GraphExtractionArtifactStorage,
        paper_repository: PaperRepository,
        ingestion_repository: IngestionRepository,
        *,
        extraction_version: str,
    ) -> None:
        self._llm = llm_provider
        self._chunk_storage = chunk_storage
        self._extraction_storage = extraction_storage
        self._paper_repo = paper_repository
        self._ingestion_repo = ingestion_repository
        self._extraction_version = extraction_version

    def extract(self, paper_id: str, paper_version_id: str | None = None) -> GraphExtractionResult:
        paper = resolve_paper(self._paper_repo, paper_id)
        version = resolve_version(self._paper_repo, paper, paper_version_id)
        chunk_document, chunk_checksum = self._validate_ready_for_extraction(paper, version)

        extraction_config_fingerprint = build_extraction_config_fingerprint(
            extraction_version=self._extraction_version,
            llm_provider=self._llm.provider_name,
            llm_model=self._llm.model_name,
            llm_provider_version=self._llm.provider_version,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            temperature=self._llm.temperature,
        )
        generation_fingerprint = build_graph_extraction_generation_fingerprint(
            chunk_artifact_checksum=chunk_checksum,
            extraction_config_fingerprint=extraction_config_fingerprint,
        )

        job = self._ingestion_repo.create_ingestion_job(
            paper_id=paper.paper_id, paper_version_id=version.paper_version_id
        )

        result, job = self._reconcile_existing_extraction(
            paper, version, job, generation_fingerprint=generation_fingerprint
        )
        if result is not None:
            return result

        return self._extract_and_finalize(
            paper,
            version,
            job,
            chunk_document=chunk_document,
            chunk_checksum=chunk_checksum,
            extraction_config_fingerprint=extraction_config_fingerprint,
            generation_fingerprint=generation_fingerprint,
        )

    # --- Validation -----------------------------------------------------

    def _validate_ready_for_extraction(
        self, paper: Paper, version: PaperVersion
    ) -> tuple[ChunkedPaperDocument, str]:
        """Chunk artifact identity/consistency check (prompt #33 step 1) --
        mirrors `VectorIndexingService._validate_ready_for_indexing`."""

        document = self._chunk_storage.try_read(
            source=paper.source, source_id=paper.source_id, version=version.version
        )
        if document is None:
            raise ChunkArtifactNotFoundError(
                f"no valid chunk artifact for {version.paper_version_id}; chunk it first"
            )
        if document.paper_id != paper.paper_id or document.paper_version_id != version.paper_version_id:
            raise GraphExtractionError(
                f"chunk artifact identity mismatch for {version.paper_version_id}"
            )
        if not document.chunks:
            raise GraphExtractionError(
                f"chunk artifact for {version.paper_version_id} has zero chunks; nothing to extract"
            )

        chunk_path = self._chunk_storage.get_path(
            source=paper.source, source_id=paper.source_id, version=version.version
        )
        return document, sha256_file(chunk_path)

    # --- Reconciliation ---------------------------------------------------

    def _reconcile_existing_extraction(
        self, paper: Paper, version: PaperVersion, job: IngestionJob, *, generation_fingerprint: str
    ) -> tuple[GraphExtractionResult | None, IngestionJob]:
        """Filesystem-first reconciliation, mirroring Prompt 6: the artifact
        on disk (re-verified against the *current* generation fingerprint)
        is the source of truth, never whatever PostgreSQL believes."""

        existing = self._extraction_storage.try_read(
            source=paper.source, source_id=paper.source_id, version=version.version
        )
        if existing is None:
            return None, self._recreate_job_if_stale(
                paper, version, job, reason="graph extraction artifact missing or corrupt"
            )

        if existing.extraction.generation_fingerprint != generation_fingerprint:
            reason = (
                "extraction generation changed "
                f"({existing.extraction.generation_fingerprint} -> {generation_fingerprint})"
            )
            logger.info(
                "existing graph extraction is stale, re-extracting paper_id=%s "
                "paper_version_id=%s reason=%r status=stale",
                paper.paper_id,
                version.paper_version_id,
                reason,
            )
            return None, self._recreate_job_if_stale(paper, version, job, reason=reason)

        # Valid, current-generation artifact exists -- reconcile PostgreSQL
        # to match it (covers "artifact finalized but the final DB write
        # failed") without calling the LLM again.
        was_already_recorded = version.graph_extraction_generation_fingerprint == generation_fingerprint
        artifact_path = self._extraction_storage.get_path(
            source=paper.source, source_id=paper.source_id, version=version.version
        )
        self._paper_repo.update_version_graph_extraction_result(
            version.paper_version_id,
            graph_extraction_artifact_path=str(artifact_path),
            graph_extracted_at=version.graph_extracted_at or datetime.now(timezone.utc),
            entity_count=len(existing.entities),
            relationship_count=len(existing.relationships),
            extraction_version=existing.extraction.version,
            extraction_config_fingerprint=existing.extraction.config_fingerprint,
            graph_extraction_generation_fingerprint=generation_fingerprint,
            graph_extraction_artifact_checksum=sha256_file(artifact_path),
        )
        job = self._advance_job_to_graph_indexing(job)

        if not was_already_recorded:
            now = datetime.now(timezone.utc)
            self._record_step(
                job.ingestion_job_id,
                self._next_attempt_number(job.ingestion_job_id),
                StepStatus.COMPLETED,
                started_at=now,
                completed_at=now,
                metadata={
                    "entity_count": len(existing.entities),
                    "relationship_count": len(existing.relationships),
                    "reconciled": True,
                },
            )

        logger.info(
            "reused existing valid graph extraction paper_id=%s paper_version_id=%s "
            "ingestion_job_id=%s status=ok",
            paper.paper_id,
            version.paper_version_id,
            job.ingestion_job_id,
        )
        return (
            GraphExtractionResult(
                job=job,
                entity_count=len(existing.entities),
                relationship_count=len(existing.relationships),
                unresolved_citation_count=len(existing.unresolved_citations),
                extraction_version=existing.extraction.version,
                extraction_config_fingerprint=existing.extraction.config_fingerprint,
                graph_extraction_generation_fingerprint=generation_fingerprint,
                extraction_reused=True,
            ),
            job,
        )

    def _recreate_job_if_stale(
        self, paper: Paper, version: PaperVersion, job: IngestionJob, *, reason: str
    ) -> IngestionJob:
        """If `job` had already reached GRAPH_INDEXING (or later) but the
        artifact backing that claim is gone/corrupt/stale, mark it failed
        and start a fresh job -- then fast-forward it to VECTOR_INDEXED,
        since the chunk source was already re-verified valid; only
        extraction needs redoing."""

        if job.status not in _STATUSES_AT_OR_PAST_GRAPH_INDEXING:
            return job

        self._ingestion_repo.mark_job_failed(
            job.ingestion_job_id, failed_stage=job.status, failure_reason=reason
        )
        logger.warning(
            "marked stale graph-extraction job failed and starting a new one paper_id=%s "
            "paper_version_id=%s ingestion_job_id=%s reason=%r status=stale",
            paper.paper_id,
            version.paper_version_id,
            job.ingestion_job_id,
            reason,
        )
        new_job = self._ingestion_repo.create_ingestion_job(
            paper_id=paper.paper_id, paper_version_id=version.paper_version_id
        )
        return self._ensure_job_at_vector_indexed(new_job)

    def _ensure_job_at_vector_indexed(self, job: IngestionJob) -> IngestionJob:
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
        if job.status == IngestionStatus.CHUNKED:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.VECTOR_INDEXING
            )
        if job.status == IngestionStatus.VECTOR_INDEXING:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.VECTOR_INDEXED
            )
        return job

    def _advance_job_to_graph_indexing(self, job: IngestionJob) -> IngestionJob:
        """No-op if `job` is already at GRAPH_INDEXING or later (mirrors
        `ChunkingService`/`VectorIndexingService`'s no-regression guard) --
        reusing a still-valid extraction on a paper that has progressed
        further (once Prompt 9 exists) must never regress its status."""

        if job.status in _STATUSES_AT_OR_PAST_GRAPH_INDEXING:
            return job
        job = self._ensure_job_at_vector_indexed(job)
        if job.status == IngestionStatus.VECTOR_INDEXED:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.GRAPH_INDEXING
            )
        return job

    # --- Extraction ---------------------------------------------------------

    def _extract_and_finalize(
        self,
        paper: Paper,
        version: PaperVersion,
        job: IngestionJob,
        *,
        chunk_document: ChunkedPaperDocument,
        chunk_checksum: str,
        extraction_config_fingerprint: str,
        generation_fingerprint: str,
    ) -> GraphExtractionResult:
        started_monotonic = time.monotonic()
        started_at = datetime.now(timezone.utc)
        attempt = self._next_attempt_number(job.ingestion_job_id)

        job = self._ensure_job_at_vector_indexed(job)
        if job.status == IngestionStatus.VECTOR_INDEXED:
            job = self._ingestion_repo.transition_job_status(
                job.ingestion_job_id, IngestionStatus.GRAPH_INDEXING
            )

        llm_calls = 0
        chunks_processed = 0
        # Optional, best-effort (prompt: DeepSeek validation, CLAUDE.md #34
        # "token_usage" observability metric) -- summed from
        # `provider.last_usage` via `getattr` so this works with any
        # provider, including ones (like `FakeLLMProvider`) that don't
        # report usage at all; never a required part of the LLMProvider
        # contract itself.
        token_usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            entities: list[ScientificEntity] = []
            relationships: list[ScientificRelationship] = []
            warnings: list[GraphExtractionWarning] = []
            unresolved_citations: list[UnresolvedCitation] = []

            # 1. Deterministic Paper/Author facts (prompt #11/#12) -- no LLM.
            paper_entity = _build_paper_entity(paper)
            entities.append(paper_entity)
            author_entities, authored_by_relationships = _build_author_facts(
                paper, paper_entity, extraction_version=self._extraction_version
            )
            entities.extend(author_entities)
            relationships.extend(authored_by_relationships)

            # 2. Deterministic citation resolution (prompt #16/#17) -- no LLM.
            for chunk in chunk_document.chunks:
                if chunk.section_type != SectionType.REFERENCES:
                    continue
                resolved_ids, chunk_unresolved = resolve_citations(
                    chunk.text, source_chunk_id=chunk.chunk_id
                )
                for target_paper_id in resolved_ids:
                    relationships.append(
                        ScientificRelationship.create(
                            source_entity_id=paper_entity.entity_id,
                            relationship_type=RelationshipType.CITES,
                            target_entity_id=target_paper_id,
                            confidence=1.0,  # deterministic resolution, not LLM-derived
                            source_chunk_id=chunk.chunk_id,
                            extraction_version=self._extraction_version,
                            metadata={"resolution": "explicit_arxiv_id"},
                        )
                    )
                for unresolved in chunk_unresolved:
                    warnings.append(
                        GraphExtractionWarning(
                            code=GraphExtractionWarningCode.CITATION_UNRESOLVED,
                            detail=unresolved.reason,
                            source_chunk_id=chunk.chunk_id,
                        )
                    )
                unresolved_citations.extend(chunk_unresolved)

            # 3. LLM semantic extraction, one chunk at a time, selected
            # sections only (prompt #6/#34).
            entity_candidates: list[ExtractedEntityCandidate] = []
            relationship_candidates: list[ExtractedRelationshipCandidate] = []
            semantic_chunks = [
                chunk
                for chunk in chunk_document.chunks
                if chunk.section_type in _SEMANTIC_EXTRACTION_SECTIONS
            ]
            for chunk in semantic_chunks:
                raw_response = self._llm.generate_structured(
                    system_prompt=build_system_prompt(),
                    user_prompt=build_user_prompt(
                        paper_id=paper.paper_id,
                        paper_title=paper.title,
                        section_type=chunk.section_type.value,
                        chunk_text=chunk.text,
                    ),
                    response_model=RawExtractionResponse,
                )
                llm_calls += 1
                chunks_processed += 1
                usage = getattr(self._llm, "last_usage", None)
                if usage is not None:
                    token_usage_totals["prompt_tokens"] += usage.prompt_tokens
                    token_usage_totals["completion_tokens"] += usage.completion_tokens
                    token_usage_totals["total_tokens"] += usage.total_tokens

                for raw_entity in raw_response.entities:
                    candidate, warning = validate_entity_candidate(
                        raw_entity, source_chunk_id=chunk.chunk_id
                    )
                    if warning is not None:
                        warnings.append(warning)
                    if candidate is not None:
                        entity_candidates.append(candidate)

                for raw_relationship in raw_response.relationships:
                    candidate, rel_warnings = validate_relationship_candidate(
                        raw_relationship, source_chunk_id=chunk.chunk_id, chunk_text=chunk.text
                    )
                    warnings.extend(rel_warnings)
                    if candidate is not None:
                        relationship_candidates.append(candidate)

            # 4. Deduplicate candidates (prompt #35), attach provenance,
            # convert to trusted domain models.
            semantic_entities, entity_lookup = _dedupe_entity_candidates(entity_candidates)
            semantic_relationships, implicit_entities = _dedupe_relationship_candidates(
                relationship_candidates,
                paper_entity=paper_entity,
                entity_lookup=entity_lookup,
                extraction_version=self._extraction_version,
            )
            entities.extend(semantic_entities)
            entities.extend(implicit_entities)
            relationships.extend(semantic_relationships)

            artifact = GraphExtractionArtifact(
                paper_id=paper.paper_id,
                paper_version_id=version.paper_version_id,
                source_chunk_artifact_checksum=chunk_checksum,
                extraction=ExtractionConfig(
                    version=self._extraction_version,
                    llm_provider=self._llm.provider_name,
                    llm_model=self._llm.model_name,
                    llm_provider_version=self._llm.provider_version,
                    prompt_version=PROMPT_VERSION,
                    schema_version=SCHEMA_VERSION,
                    temperature=self._llm.temperature,
                    config_fingerprint=extraction_config_fingerprint,
                    generation_fingerprint=generation_fingerprint,
                ),
                entities=entities,
                relationships=relationships,
                unresolved_citations=unresolved_citations,
                warnings=warnings,
            )
            artifact_path = self._extraction_storage.write(
                artifact, source=paper.source, source_id=paper.source_id, version=version.version
            )
            artifact_checksum = sha256_file(artifact_path)
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
                metadata={
                    "chunks_processed": chunks_processed,
                    "llm_calls": llm_calls,
                    "duration_ms": duration_ms,
                    "token_usage": token_usage_totals,
                },
            )
            self._ingestion_repo.mark_job_failed(
                job.ingestion_job_id,
                failed_stage=IngestionStatus.GRAPH_INDEXING,
                failure_reason=str(exc)[:2000],
            )
            logger.warning(
                "graph extraction failed paper_id=%s paper_version_id=%s ingestion_job_id=%s "
                "attempt=%d chunks_processed=%d llm_calls=%d duration_ms=%d status=error error=%s",
                paper.paper_id,
                version.paper_version_id,
                job.ingestion_job_id,
                attempt,
                chunks_processed,
                llm_calls,
                duration_ms,
                exc,
            )
            raise

        self._paper_repo.update_version_graph_extraction_result(
            version.paper_version_id,
            graph_extraction_artifact_path=str(artifact_path),
            graph_extracted_at=datetime.now(timezone.utc),
            entity_count=len(artifact.entities),
            relationship_count=len(artifact.relationships),
            extraction_version=self._extraction_version,
            extraction_config_fingerprint=extraction_config_fingerprint,
            graph_extraction_generation_fingerprint=generation_fingerprint,
            graph_extraction_artifact_checksum=artifact_checksum,
        )
        # Job stays at GRAPH_INDEXING -- GRAPH_INDEXED is never claimed
        # here (prompt #31): that would falsely imply Neo4j indexing
        # happened. `job` is already at GRAPH_INDEXING from above.

        duration_ms = int((time.monotonic() - started_monotonic) * 1000)
        self._record_step(
            job.ingestion_job_id,
            attempt,
            StepStatus.COMPLETED,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            metadata={
                "chunks_processed": chunks_processed,
                "entity_count": len(artifact.entities),
                "relationship_count": len(artifact.relationships),
                "llm_calls": llm_calls,
                "duration_ms": duration_ms,
                "token_usage": token_usage_totals,
            },
        )
        logger.info(
            "graph extraction succeeded paper_id=%s paper_version_id=%s ingestion_job_id=%s "
            "llm_provider=%s llm_model=%s chunks_processed=%d entity_count=%d "
            "relationship_count=%d llm_calls=%d duration_ms=%d total_tokens=%d "
            "extraction_reused=False status=ok",
            paper.paper_id,
            version.paper_version_id,
            job.ingestion_job_id,
            self._llm.provider_name,
            self._llm.model_name,
            chunks_processed,
            len(artifact.entities),
            len(artifact.relationships),
            llm_calls,
            duration_ms,
            token_usage_totals["total_tokens"],
        )
        return GraphExtractionResult(
            job=job,
            entity_count=len(artifact.entities),
            relationship_count=len(artifact.relationships),
            unresolved_citation_count=len(artifact.unresolved_citations),
            extraction_version=self._extraction_version,
            extraction_config_fingerprint=extraction_config_fingerprint,
            graph_extraction_generation_fingerprint=generation_fingerprint,
            extraction_reused=False,
        )

    # --- Steps -------------------------------------------------------------

    def _next_attempt_number(self, ingestion_job_id: str) -> int:
        existing = [
            step
            for step in self._ingestion_repo.list_steps(ingestion_job_id)
            if step.stage == ProcessingStage.GRAPH_EXTRACT
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
                stage=ProcessingStage.GRAPH_EXTRACT,
                status=status,
                attempt=attempt,
                started_at=started_at,
                completed_at=completed_at,
                error_code=error_code,
                error_message=error_message,
                metadata=metadata or {},
            )
        )


# --- Deterministic entity/relationship builders --------------------------


def _build_paper_entity(paper: Paper) -> ScientificEntity:
    """The current paper's PAPER entity (prompt #12) -- trusted identity:
    `entity_id` is the paper's own `paper_id`, never a name-hash. This is
    what keeps this graph node joinable back to its PostgreSQL/Qdrant
    records without ambiguity, and is why `ScientificEntity.create()`'s
    normal name-hash derivation is deliberately bypassed here."""

    return ScientificEntity(
        entity_id=paper.paper_id,
        entity_type=EntityType.PAPER,
        canonical_name=paper.title,
        metadata={"source": paper.source, "source_id": paper.source_id},
    )


def _build_author_facts(
    paper: Paper, paper_entity: ScientificEntity, *, extraction_version: str
) -> tuple[list[ScientificEntity], list[ScientificRelationship]]:
    """`Paper -AUTHORED_BY-> Author` facts from trusted arXiv metadata
    (prompt #11) -- no LLM involved."""

    entities: list[ScientificEntity] = []
    relationships: list[ScientificRelationship] = []
    seen: set[str] = set()
    for author_name in paper.authors:
        normalized = normalize_whitespace(author_name)
        if not normalized:
            continue
        key = normalize_identity_key(normalized)
        if key in seen:
            continue
        seen.add(key)

        author_entity = ScientificEntity.create(
            entity_type=EntityType.AUTHOR, canonical_name=normalized
        )
        entities.append(author_entity)
        relationships.append(
            ScientificRelationship.create(
                source_entity_id=paper_entity.entity_id,
                relationship_type=RelationshipType.AUTHORED_BY,
                target_entity_id=author_entity.entity_id,
                confidence=1.0,
                extraction_version=extraction_version,
                metadata={"resolution": "arxiv_metadata"},
            )
        )
    return entities, relationships


# Bounded so a burst of near-identical quotes never bloats one entity's
# metadata unreasonably (prompt #18: "do not store huge... redundantly").
_MAX_STORED_EVIDENCE_QUOTES = 5


def _dedupe_entity_candidates(
    candidates: list[ExtractedEntityCandidate],
) -> tuple[list[ScientificEntity], dict[tuple[EntityType, str], ScientificEntity]]:
    """Group by (entity_type, normalized name) -- conservative aggregation
    only (prompt #35/#36): this is *not* cross-paper alias resolution
    ("Graph RAG" vs "GraphRAG" vs "Graph-based RAG" staying distinct
    candidates is correct here), just collapsing the same literal name's
    repeated mentions within this one paper's extraction. Confidence is
    the max across duplicates; every supporting chunk id is preserved in
    metadata (`ScientificRelationship`/`ScientificEntity`'s single
    `source_chunk_id`/lack thereof isn't enough on its own, prompt #35's
    "preserve multiple source chunk IDs" requirement)."""

    groups: dict[tuple[EntityType, str], list[ExtractedEntityCandidate]] = defaultdict(list)
    for candidate in candidates:
        key = (candidate.entity_type, normalize_identity_key(candidate.name))
        groups[key].append(candidate)

    entities: list[ScientificEntity] = []
    lookup: dict[tuple[EntityType, str], ScientificEntity] = {}
    for key, group in groups.items():
        entity_type, _normalized_key = key
        canonical_name = group[0].name  # first-seen surface form
        aliases: list[str] = []
        for candidate in group:
            aliases.extend(candidate.aliases)
        chunk_ids = sorted({candidate.source_chunk_id for candidate in group})
        evidence_quotes = [c.evidence_quote for c in group if c.evidence_quote][
            :_MAX_STORED_EVIDENCE_QUOTES
        ]
        entity = ScientificEntity.create(
            entity_type=entity_type,
            canonical_name=canonical_name,
            aliases=aliases,
            metadata={
                "extraction_confidence": max(c.confidence for c in group),
                "source_chunk_ids": chunk_ids,
                "evidence_quotes": evidence_quotes,
            },
        )
        entities.append(entity)
        lookup[key] = entity
    return entities, lookup


def _dedupe_relationship_candidates(
    candidates: list[ExtractedRelationshipCandidate],
    *,
    paper_entity: ScientificEntity,
    entity_lookup: dict[tuple[EntityType, str], ScientificEntity],
    extraction_version: str,
) -> tuple[list[ScientificRelationship], list[ScientificEntity]]:
    """Group by (source, relationship_type, target) -- aggregating multiple
    supporting chunks per relation (prompt #35). The source of every
    semantic relationship is always the current paper's own trusted
    `paper_entity.entity_id`, never derived from the LLM-reported
    `source_name` text (prompt #10: never let the LLM invent source paper
    identity). A target entity not already present from the entity list
    (the LLM mentioned it only via a relationship) is created here as an
    implicit mention, added to the shared lookup so later candidates
    referencing the same name reuse it rather than duplicating it."""

    entity_lookup = dict(entity_lookup)  # local copy -- never mutate the caller's dict
    implicit_entities: list[ScientificEntity] = []
    groups: dict[tuple[str, RelationshipType, str], list[ExtractedRelationshipCandidate]] = (
        defaultdict(list)
    )

    for candidate in candidates:
        target_key = (candidate.target_type, normalize_identity_key(candidate.target_name))
        target_entity = entity_lookup.get(target_key)
        if target_entity is None:
            target_entity = ScientificEntity.create(
                entity_type=candidate.target_type,
                canonical_name=candidate.target_name,
                metadata={"extraction_confidence": candidate.confidence, "implicit": True},
            )
            entity_lookup[target_key] = target_entity
            implicit_entities.append(target_entity)

        group_key = (paper_entity.entity_id, candidate.relationship_type, target_entity.entity_id)
        groups[group_key].append(candidate)

    relationships: list[ScientificRelationship] = []
    for (source_id, relationship_type, target_id), group in groups.items():
        chunk_ids = sorted({candidate.source_chunk_id for candidate in group})
        evidence_quotes = [c.evidence_quote for c in group if c.evidence_quote][
            :_MAX_STORED_EVIDENCE_QUOTES
        ]
        relationships.append(
            ScientificRelationship.create(
                source_entity_id=source_id,
                relationship_type=relationship_type,
                target_entity_id=target_id,
                confidence=max(candidate.confidence for candidate in group),
                source_chunk_id=chunk_ids[0],
                extraction_version=extraction_version,
                metadata={"supporting_chunk_ids": chunk_ids, "evidence_quotes": evidence_quotes},
            )
        )
    return relationships, implicit_entities
