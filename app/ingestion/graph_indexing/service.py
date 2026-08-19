"""Graph indexing service: orchestrates artifact validation, canonical
entity resolution, batched Neo4j writes, and idempotency/reconciliation
for the graph-indexing stage.

Target flow (prompt #37)::

    GRAPH_INDEXING -> validate graph_extraction.json
    -> build citation-target entities (real or placeholder Paper nodes)
    -> canonicalize (CanonicalEntityResolver, no LLM)
    -> ensure Neo4j schema -> batch upsert entities -> batch upsert
       relationships -> verify exact relationship-id set -> delete stale
       prior-generation relationships for this paper version
    -> persist Postgres metadata -> GRAPH_INDEXED

Never calls an LLM (prompt #2/#10) -- canonicalization is a pure function
of the already-validated extraction artifact plus the deterministic
`EntityAliasRegistry`.

Reconciliation mirrors Prompt 7's Qdrant-first design, applied to Neo4j:
the *exact set* of relationship ids currently tagged with the *current*
`graph_index_generation_fingerprint` for this paper version (prompt #50 --
not a bare count, since shared nodes make a global count meaningless) is
the source of truth for "has this already been indexed", never whatever
PostgreSQL currently records.

Node deletion is deliberately never implemented (prompt #47): a canonical
entity node (Method/Dataset/Task/Author/Paper) is shared across papers and
is never deleted by this service, even when every relationship pointing
at it from one paper is removed during stale-generation cleanup. Harmless
orphans are preferred over any risk to a shared node still referenced by
another paper's still-current generation.
"""

import logging
import time
from datetime import datetime, timezone

from pydantic import BaseModel

from app.core.exceptions import GraphExtractionArtifactNotFoundError, GraphIndexingError
from app.domain.enums import EntityType, IngestionStatus, ProcessingStage, RelationshipType, StepStatus
from app.domain.ingestion import TERMINAL_STATUSES, IngestionJob, IngestionStepState
from app.domain.knowledge import ScientificEntity
from app.domain.papers import Paper, PaperVersion
from app.graph.repository import GraphRepository
from app.ingestion.canonical_resolution.alias_registry import EntityAliasRegistry, get_default_alias_registry
from app.ingestion.canonical_resolution.fingerprint import (
    CANONICAL_ONTOLOGY_VERSION,
    NORMALIZATION_ALGORITHM_VERSION,
    build_canonicalization_config_fingerprint,
    build_graph_index_generation_fingerprint,
)
from app.ingestion.canonical_resolution.models import CanonicalGraph
from app.ingestion.canonical_resolution.resolver import CanonicalEntityResolver
from app.ingestion.checksums import sha256_file
from app.ingestion.graph_extraction.models import GraphExtractionArtifact
from app.ingestion.graph_extraction.storage import GraphExtractionArtifactStorage
from app.ingestion.graph_indexing.graph_mapping import entity_to_node_input, relationship_to_relationship_input
from app.ingestion.paper_resolution import resolve_paper, resolve_version
from app.storage.postgres.repositories.ingestion import IngestionRepository
from app.storage.postgres.repositories.papers import PaperRepository

logger = logging.getLogger(__name__)

# Every status a job can hold *at or after* GRAPH_INDEXED. READY is not
# actually reachable yet by anything in this codebase (nothing transitions
# a job to READY), but is included proactively anyway -- the same defensive
# pattern that caught real bugs twice before (Prompt 7's ChunkingService,
# Prompt 8's VectorIndexingService). Unlike those two prior cases, READY
# is *terminal* (`TERMINAL_STATUSES`), so `_recreate_job_if_stale` below
# must not blindly call `mark_job_failed` on it -- a job already at a
# terminal status can never legally transition to FAILED.
_STATUSES_AT_OR_PAST_GRAPH_INDEXED = frozenset(
    {IngestionStatus.GRAPH_INDEXED, IngestionStatus.READY}
)


class GraphIndexResult(BaseModel):
    """Outcome of `GraphIndexingService.index_paper_version()`."""

    job: IngestionJob
    canonical_entity_count: int
    graph_relationship_count: int
    canonicalization_config_fingerprint: str
    graph_index_generation_fingerprint: str
    unresolved_citations_skipped: int
    index_reused: bool = False


class GraphIndexingService:
    """Explicit graph indexing: the only thing FastAPI routes should call
    for `POST /api/v1/papers/{paper_id}/graph-index`."""

    def __init__(
        self,
        graph_repository: GraphRepository,
        extraction_storage: GraphExtractionArtifactStorage,
        paper_repository: PaperRepository,
        ingestion_repository: IngestionRepository,
        *,
        canonicalization_version: str,
        neo4j_database: str,
        alias_registry: EntityAliasRegistry | None = None,
    ) -> None:
        self._graph_repo = graph_repository
        self._extraction_storage = extraction_storage
        self._paper_repo = paper_repository
        self._ingestion_repo = ingestion_repository
        self._canonicalization_version = canonicalization_version
        self._neo4j_database = neo4j_database
        self._alias_registry = alias_registry or get_default_alias_registry()
        self._resolver = CanonicalEntityResolver(self._alias_registry)

    def index_paper_version(
        self, paper_id: str, paper_version_id: str | None = None
    ) -> GraphIndexResult:
        paper = resolve_paper(self._paper_repo, paper_id)
        version = resolve_version(self._paper_repo, paper, paper_version_id)
        artifact, artifact_checksum = self._validate_ready_for_indexing(paper, version)

        entities = self._prepare_entities(artifact, paper)

        canonicalization_config_fingerprint = build_canonicalization_config_fingerprint(
            canonicalization_version=self._canonicalization_version,
            normalization_algorithm_version=NORMALIZATION_ALGORITHM_VERSION,
            alias_registry_version=self._alias_registry.version,
            alias_registry_checksum=self._alias_registry.checksum,
            ontology_version=CANONICAL_ONTOLOGY_VERSION,
        )
        generation_fingerprint = build_graph_index_generation_fingerprint(
            graph_extraction_artifact_checksum=artifact_checksum,
            canonicalization_config_fingerprint=canonicalization_config_fingerprint,
        )

        canonical_graph = self._resolver.build_canonical_graph(
            paper_id=paper.paper_id,
            paper_version_id=version.paper_version_id,
            entities=entities,
            relationships=artifact.relationships,
            canonicalization_version=self._canonicalization_version,
            canonicalization_config_fingerprint=canonicalization_config_fingerprint,
            graph_index_generation_fingerprint=generation_fingerprint,
        )

        # Idempotent, cheap -- safe to call on every request (prompt #26).
        self._graph_repo.ensure_schema()

        job = self._ingestion_repo.create_ingestion_job(
            paper_id=paper.paper_id, paper_version_id=version.paper_version_id
        )

        result, job = self._reconcile_existing_index(
            paper, version, job, canonical_graph=canonical_graph, generation_fingerprint=generation_fingerprint
        )
        if result is not None:
            return result

        return self._index_and_finalize(
            paper,
            version,
            job,
            canonical_graph=canonical_graph,
            generation_fingerprint=generation_fingerprint,
            unresolved_citation_count=len(artifact.unresolved_citations),
        )

    # --- Validation -----------------------------------------------------

    def _validate_ready_for_indexing(
        self, paper: Paper, version: PaperVersion
    ) -> tuple[GraphExtractionArtifact, str]:
        """Extraction-artifact identity/consistency checks before any
        canonicalization happens (mirrors `VectorIndexingService`'s
        equivalent chunk-artifact checks)."""

        artifact = self._extraction_storage.try_read(
            source=paper.source, source_id=paper.source_id, version=version.version
        )
        if artifact is None:
            raise GraphExtractionArtifactNotFoundError(
                f"no valid graph extraction artifact for {version.paper_version_id}; extract it first"
            )
        if artifact.paper_id != paper.paper_id or artifact.paper_version_id != version.paper_version_id:
            raise GraphIndexingError(
                f"graph extraction artifact identity mismatch for {version.paper_version_id}"
            )
        if not any(
            entity.entity_type == EntityType.PAPER and entity.entity_id == paper.paper_id
            for entity in artifact.entities
        ):
            raise GraphIndexingError(
                f"graph extraction artifact for {version.paper_version_id} is missing its own "
                "PAPER entity; never ingest raw/incomplete extraction output into Neo4j"
            )

        artifact_path = self._extraction_storage.get_path(
            source=paper.source, source_id=paper.source_id, version=version.version
        )
        return artifact, sha256_file(artifact_path)

    # --- Preparation (never LLM, may read PostgreSQL) ----------------------

    def _prepare_entities(
        self, artifact: GraphExtractionArtifact, paper: Paper
    ) -> list[ScientificEntity]:
        """Everything the resolver needs but can't build itself: enriching
        this paper's own PAPER entity with trusted metadata not present in
        `graph_extraction.json`, and materializing a real-or-placeholder
        Paper entity for every `CITES` target (prompt #19) -- the resolver
        (`app.ingestion.canonical_resolution`) is deliberately kept free of
        PostgreSQL access, so this preparation step lives here instead."""

        entities = [self._enrich_if_self_paper(entity, paper) for entity in artifact.entities]
        existing_ids = {entity.entity_id for entity in entities}

        for relationship in artifact.relationships:
            if relationship.relationship_type != RelationshipType.CITES:
                continue
            target_id = relationship.target_entity_id
            if target_id in existing_ids:
                continue
            entities.append(self._build_citation_target_entity(target_id))
            existing_ids.add(target_id)

        return entities

    def _enrich_if_self_paper(self, entity: ScientificEntity, paper: Paper) -> ScientificEntity:
        if entity.entity_type != EntityType.PAPER or entity.entity_id != paper.paper_id:
            return entity
        if paper.published_at is None:
            return entity
        metadata = dict(entity.metadata)
        metadata["published_at"] = paper.published_at.isoformat()
        return entity.model_copy(update={"metadata": metadata})

    def _build_citation_target_entity(self, target_paper_id: str) -> ScientificEntity:
        """A trusted `Paper` node for a `CITES` target -- real (this
        paper's own PostgreSQL record) if already discovered, otherwise a
        minimal reference-only placeholder (prompt #19). Never fabricates
        a title/authors/abstract for a paper that hasn't actually been
        discovered."""

        known = self._paper_repo.get_by_paper_id(target_paper_id)
        if known is not None:
            return ScientificEntity(
                entity_id=known.paper_id,
                entity_type=EntityType.PAPER,
                canonical_name=known.title,
                metadata={
                    "source": known.source,
                    "source_id": known.source_id,
                    "published_at": known.published_at.isoformat() if known.published_at else None,
                },
            )

        _prefix, source, source_id = target_paper_id.split(":", 2)
        return ScientificEntity(
            entity_id=target_paper_id,
            entity_type=EntityType.PAPER,
            canonical_name=f"{source}:{source_id}",
            metadata={"source": source, "source_id": source_id, "placeholder": True},
        )

    # --- Reconciliation ---------------------------------------------------

    def _reconcile_existing_index(
        self,
        paper: Paper,
        version: PaperVersion,
        job: IngestionJob,
        *,
        canonical_graph: CanonicalGraph,
        generation_fingerprint: str,
    ) -> tuple[GraphIndexResult | None, IngestionJob]:
        """Neo4j-first reconciliation (prompt #49/#50): the *exact set* of
        current-generation relationship ids, plus the paper's own node
        actually existing, is what "already indexed" means -- never a bare
        count (a paper with zero relationships would otherwise trivially
        "match" an empty Neo4j), and never whatever PostgreSQL believes.
        """

        expected_ids = {r.relationship_id for r in canonical_graph.relationships}
        actual_ids = self._graph_repo.get_relationship_ids_for_generation(
            version.paper_version_id, generation_fingerprint=generation_fingerprint
        )
        paper_node_exists = self._graph_repo.get_entity(paper.paper_id) is not None

        if actual_ids != expected_ids or not paper_node_exists:
            if not paper_node_exists:
                reason = "paper node missing from Neo4j"
            elif not actual_ids:
                reason = "no current-generation relationships found in Neo4j"
            else:
                reason = (
                    f"partial/stale current-generation relationships in Neo4j "
                    f"({len(actual_ids)}/{len(expected_ids)})"
                )
            logger.info(
                "existing graph index is stale/incomplete, reindexing paper_id=%s "
                "paper_version_id=%s reason=%r status=stale",
                paper.paper_id,
                version.paper_version_id,
                reason,
            )
            return None, self._recreate_job_if_stale(paper, version, job, reason=reason)

        # Valid, current-generation graph is already fully present --
        # reconcile PostgreSQL to match (covers "Neo4j complete but the
        # final DB write failed") without rewriting anything.
        was_already_recorded = version.graph_index_generation_fingerprint == generation_fingerprint
        self._paper_repo.update_version_graph_index_result(
            version.paper_version_id,
            graph_indexed_at=version.graph_indexed_at or datetime.now(timezone.utc),
            canonical_entity_count=len(canonical_graph.entities),
            graph_relationship_count=len(canonical_graph.relationships),
            canonicalization_config_fingerprint=canonical_graph.canonicalization_config_fingerprint,
            graph_index_generation_fingerprint=generation_fingerprint,
            neo4j_database=self._neo4j_database,
        )
        job = self._advance_job_to_graph_indexed(job)

        if not was_already_recorded:
            now = datetime.now(timezone.utc)
            self._record_step(
                job.ingestion_job_id,
                self._next_attempt_number(job.ingestion_job_id),
                StepStatus.COMPLETED,
                started_at=now,
                completed_at=now,
                metadata={
                    "canonical_entity_count": len(canonical_graph.entities),
                    "graph_relationship_count": len(canonical_graph.relationships),
                    "reconciled": True,
                },
            )

        logger.info(
            "reused existing valid graph index paper_id=%s paper_version_id=%s "
            "ingestion_job_id=%s status=ok",
            paper.paper_id,
            version.paper_version_id,
            job.ingestion_job_id,
        )
        return (
            GraphIndexResult(
                job=job,
                canonical_entity_count=len(canonical_graph.entities),
                graph_relationship_count=len(canonical_graph.relationships),
                canonicalization_config_fingerprint=canonical_graph.canonicalization_config_fingerprint,
                graph_index_generation_fingerprint=generation_fingerprint,
                unresolved_citations_skipped=0,  # not recomputed on the reuse path; see artifact for the real count
                index_reused=True,
            ),
            job,
        )

    def _recreate_job_if_stale(
        self, paper: Paper, version: PaperVersion, job: IngestionJob, *, reason: str
    ) -> IngestionJob:
        """If `job` had already reached GRAPH_INDEXED (or, defensively,
        READY) but Neo4j no longer backs that claim, start a fresh job --
        then fast-forward it to GRAPH_INDEXING. `mark_job_failed` is only
        called when `job.status` is not itself terminal: READY is
        included in `_STATUSES_AT_OR_PAST_GRAPH_INDEXED` purely
        defensively (nothing currently transitions a job to READY), but a
        job already at a terminal status can never legally transition to
        FAILED -- calling `mark_job_failed` on one would raise
        `InvalidIngestionTransitionError` instead of recovering."""

        if job.status not in _STATUSES_AT_OR_PAST_GRAPH_INDEXED:
            return job

        if job.status not in TERMINAL_STATUSES:
            self._ingestion_repo.mark_job_failed(
                job.ingestion_job_id, failed_stage=job.status, failure_reason=reason
            )
        logger.warning(
            "marked stale graph-index job failed and starting a new one paper_id=%s "
            "paper_version_id=%s ingestion_job_id=%s reason=%r status=stale",
            paper.paper_id,
            version.paper_version_id,
            job.ingestion_job_id,
            reason,
        )
        new_job = self._ingestion_repo.create_ingestion_job(
            paper_id=paper.paper_id, paper_version_id=version.paper_version_id
        )
        return self._ensure_job_at_graph_indexing(new_job)

    def _ensure_job_at_graph_indexing(self, job: IngestionJob) -> IngestionJob:
        if job.status == IngestionStatus.DISCOVERED:
            job = self._ingestion_repo.transition_job_status(job.ingestion_job_id, IngestionStatus.DOWNLOADING)
        if job.status == IngestionStatus.DOWNLOADING:
            job = self._ingestion_repo.transition_job_status(job.ingestion_job_id, IngestionStatus.DOWNLOADED)
        if job.status == IngestionStatus.DOWNLOADED:
            job = self._ingestion_repo.transition_job_status(job.ingestion_job_id, IngestionStatus.PARSING)
        if job.status == IngestionStatus.PARSING:
            job = self._ingestion_repo.transition_job_status(job.ingestion_job_id, IngestionStatus.PARSED)
        if job.status == IngestionStatus.PARSED:
            job = self._ingestion_repo.transition_job_status(job.ingestion_job_id, IngestionStatus.CHUNKING)
        if job.status == IngestionStatus.CHUNKING:
            job = self._ingestion_repo.transition_job_status(job.ingestion_job_id, IngestionStatus.CHUNKED)
        if job.status == IngestionStatus.CHUNKED:
            job = self._ingestion_repo.transition_job_status(job.ingestion_job_id, IngestionStatus.VECTOR_INDEXING)
        if job.status == IngestionStatus.VECTOR_INDEXING:
            job = self._ingestion_repo.transition_job_status(job.ingestion_job_id, IngestionStatus.VECTOR_INDEXED)
        if job.status == IngestionStatus.VECTOR_INDEXED:
            job = self._ingestion_repo.transition_job_status(job.ingestion_job_id, IngestionStatus.GRAPH_INDEXING)
        return job

    def _advance_job_to_graph_indexed(self, job: IngestionJob) -> IngestionJob:
        """No-op if `job` is already at GRAPH_INDEXED (or later) -- reusing
        a still-valid graph generation must never regress a job's reported
        status (mirrors `VectorIndexingService._advance_job_to_vector_indexed`)."""

        if job.status in _STATUSES_AT_OR_PAST_GRAPH_INDEXED:
            return job
        job = self._ensure_job_at_graph_indexing(job)
        if job.status == IngestionStatus.GRAPH_INDEXING:
            job = self._ingestion_repo.transition_job_status(job.ingestion_job_id, IngestionStatus.GRAPH_INDEXED)
        return job

    # --- Indexing -----------------------------------------------------------

    def _index_and_finalize(
        self,
        paper: Paper,
        version: PaperVersion,
        job: IngestionJob,
        *,
        canonical_graph: CanonicalGraph,
        generation_fingerprint: str,
        unresolved_citation_count: int,
    ) -> GraphIndexResult:
        started_monotonic = time.monotonic()
        started_at = datetime.now(timezone.utc)
        attempt = self._next_attempt_number(job.ingestion_job_id)

        job = self._ensure_job_at_graph_indexing(job)

        try:
            node_inputs = [entity_to_node_input(entity) for entity in canonical_graph.entities]
            relationship_inputs = [
                relationship_to_relationship_input(
                    relationship,
                    paper_version_id=version.paper_version_id,
                    graph_index_generation_fingerprint=generation_fingerprint,
                )
                for relationship in canonical_graph.relationships
            ]

            self._graph_repo.upsert_entities(node_inputs)
            self._graph_repo.upsert_relationships(relationship_inputs)

            # Verify before trusting the write (prompt #50): exact
            # relationship-id set, not a bare count -- a shared target
            # node existing already would otherwise make a count-only
            # check pass even if this paper's own edges never landed.
            expected_ids = {relationship.relationship_id for relationship in canonical_graph.relationships}
            actual_ids = self._graph_repo.get_relationship_ids_for_generation(
                version.paper_version_id, generation_fingerprint=generation_fingerprint
            )
            if actual_ids != expected_ids:
                missing = expected_ids - actual_ids
                raise GraphIndexingError(
                    f"post-upsert verification failed for {version.paper_version_id}: expected "
                    f"{len(expected_ids)} current-generation relationships, Neo4j has "
                    f"{len(actual_ids)} ({len(missing)} missing)"
                )

            # Only now -- generation verified complete -- remove this
            # paper version's previous-generation relationships. Shared
            # canonical nodes are never touched (prompt #45/#46/#47).
            deleted = self._graph_repo.delete_generation(
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
                failed_stage=IngestionStatus.GRAPH_INDEXING,
                failure_reason=str(exc)[:2000],
            )
            logger.warning(
                "graph indexing failed paper_id=%s paper_version_id=%s ingestion_job_id=%s "
                "attempt=%d duration_ms=%d status=error error=%s",
                paper.paper_id,
                version.paper_version_id,
                job.ingestion_job_id,
                attempt,
                duration_ms,
                exc,
            )
            raise

        self._paper_repo.update_version_graph_index_result(
            version.paper_version_id,
            graph_indexed_at=datetime.now(timezone.utc),
            canonical_entity_count=len(canonical_graph.entities),
            graph_relationship_count=len(canonical_graph.relationships),
            canonicalization_config_fingerprint=canonical_graph.canonicalization_config_fingerprint,
            graph_index_generation_fingerprint=generation_fingerprint,
            neo4j_database=self._neo4j_database,
        )
        job = self._ingestion_repo.transition_job_status(job.ingestion_job_id, IngestionStatus.GRAPH_INDEXED)

        duration_ms = int((time.monotonic() - started_monotonic) * 1000)
        self._record_step(
            job.ingestion_job_id,
            attempt,
            StepStatus.COMPLETED,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            metadata={
                "canonical_entity_count": len(canonical_graph.entities),
                "graph_relationship_count": len(canonical_graph.relationships),
                "stale_relationships_deleted": deleted,
                "duration_ms": duration_ms,
            },
        )
        logger.info(
            "graph indexing succeeded paper_id=%s paper_version_id=%s ingestion_job_id=%s "
            "canonical_entity_count=%d graph_relationship_count=%d stale_relationships_deleted=%d "
            "duration_ms=%d index_reused=False status=ok",
            paper.paper_id,
            version.paper_version_id,
            job.ingestion_job_id,
            len(canonical_graph.entities),
            len(canonical_graph.relationships),
            deleted,
            duration_ms,
        )
        return GraphIndexResult(
            job=job,
            canonical_entity_count=len(canonical_graph.entities),
            graph_relationship_count=len(canonical_graph.relationships),
            canonicalization_config_fingerprint=canonical_graph.canonicalization_config_fingerprint,
            graph_index_generation_fingerprint=generation_fingerprint,
            unresolved_citations_skipped=unresolved_citation_count,
            index_reused=False,
        )

    # --- Steps -------------------------------------------------------------

    def _next_attempt_number(self, ingestion_job_id: str) -> int:
        existing = [
            step
            for step in self._ingestion_repo.list_steps(ingestion_job_id)
            if step.stage == ProcessingStage.GRAPH_INDEX
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
                stage=ProcessingStage.GRAPH_INDEX,
                status=status,
                attempt=attempt,
                started_at=started_at,
                completed_at=completed_at,
                error_code=error_code,
                error_message=error_message,
                metadata=metadata or {},
            )
        )
