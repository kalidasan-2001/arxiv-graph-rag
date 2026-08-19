"""Paper and paper-version persistence.

Idempotency policy (CLAUDE.md #13; prompt #12-13):

* `upsert_paper` -- an existing `(source, source_id)` pair updates its
  mutable metadata fields in place; it never creates a duplicate row, and
  `paper_id` (the stable domain identity) never changes.
* `get_or_create_paper_version` -- an existing `(paper_id, version)` pair
  is returned as-is, not recreated. If the caller's `checksum` differs
  from the one already stored, that is treated as a *conflict*, not a
  silent overwrite: a recorded paper version is immutable content, per the
  prompt's explicit instruction not to silently overwrite unexpected
  changes. Callers that legitimately need to reprocess should do so via an
  explicit future "force reindex" path (prompt #15), not by mutating an
  existing version's identity.
"""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import (
    PaperNotFoundError,
    PaperVersionNotFoundError,
    PersistenceConflictError,
)
from app.domain.papers import Paper, PaperVersion
from app.storage.postgres.mappings import (
    domain_paper_to_record,
    domain_paper_version_to_record,
    paper_record_to_domain,
    paper_version_record_to_domain,
    update_paper_record_from_domain,
)
from app.storage.postgres.models import PaperRecord, PaperVersionRecord


class PaperRepository:
    """Persistence operations for logical papers and their versions."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_paper_id(self, paper_id: str) -> Paper | None:
        record = self._session.get(PaperRecord, paper_id)
        return paper_record_to_domain(record) if record else None

    def require_by_paper_id(self, paper_id: str) -> Paper:
        paper = self.get_by_paper_id(paper_id)
        if paper is None:
            raise PaperNotFoundError(f"paper {paper_id} not found")
        return paper

    def get_by_source(self, source: str, source_id: str) -> Paper | None:
        stmt = select(PaperRecord).where(
            PaperRecord.source == source, PaperRecord.source_id == source_id
        )
        record = self._session.execute(stmt).scalar_one_or_none()
        return paper_record_to_domain(record) if record else None

    def upsert_paper(self, paper: Paper) -> Paper:
        """Insert a new paper, or update mutable metadata on an existing one."""

        existing = self._session.get(PaperRecord, paper.paper_id)
        if existing is None:
            record = domain_paper_to_record(paper)
            self._session.add(record)
            self._session.flush()
            return paper_record_to_domain(record)

        update_paper_record_from_domain(existing, paper)
        self._session.flush()
        return paper_record_to_domain(existing)

    def get_paper_version(self, paper_version_id: str) -> PaperVersion | None:
        record = self._session.get(PaperVersionRecord, paper_version_id)
        return paper_version_record_to_domain(record) if record else None

    def require_paper_version(self, paper_version_id: str) -> PaperVersion:
        version = self.get_paper_version(paper_version_id)
        if version is None:
            raise PaperVersionNotFoundError(f"paper version {paper_version_id} not found")
        return version

    def list_versions(self, paper_id: str) -> list[PaperVersion]:
        """All discovered versions of a logical paper, in no particular order."""

        stmt = select(PaperVersionRecord).where(PaperVersionRecord.paper_id == paper_id)
        records = self._session.execute(stmt).scalars().all()
        return [paper_version_record_to_domain(record) for record in records]

    def get_or_create_paper_version(self, version: PaperVersion) -> PaperVersion:
        """Return the existing `(paper_id, version)` record, or create it.

        Raises `PersistenceConflictError` if a version already exists with
        a different, non-null `checksum` than the one provided.
        """

        stmt = select(PaperVersionRecord).where(
            PaperVersionRecord.paper_id == version.paper_id,
            PaperVersionRecord.version == version.version,
        )
        existing = self._session.execute(stmt).scalar_one_or_none()
        if existing is None:
            record = domain_paper_version_to_record(version)
            self._session.add(record)
            self._session.flush()
            return paper_version_record_to_domain(record)

        if (
            version.checksum is not None
            and existing.checksum is not None
            and version.checksum != existing.checksum
        ):
            raise PersistenceConflictError(
                f"paper version {existing.paper_version_id} already exists with a "
                "different checksum; a changed checksum for an existing version is "
                "treated as a conflict, not a silent overwrite"
            )
        return paper_version_record_to_domain(existing)

    def update_version_artifact(
        self,
        paper_version_id: str,
        *,
        checksum: str,
        storage_path: str,
        file_size_bytes: int,
        downloaded_at: datetime,
    ) -> PaperVersion:
        """Persist raw-PDF artifact metadata onto an existing paper version.

        Unlike `get_or_create_paper_version`, this always overwrites the
        given fields in place -- it is only ever called by
        `PdfAcquisitionService` after a real download or filesystem
        reconciliation has determined the correct values, so there's no
        create-vs-conflict ambiguity to guard against here.
        """

        record = self._session.get(PaperVersionRecord, paper_version_id)
        if record is None:
            raise PaperVersionNotFoundError(f"paper version {paper_version_id} not found")

        record.checksum = checksum
        record.storage_path = storage_path
        record.file_size_bytes = file_size_bytes
        record.downloaded_at = downloaded_at
        self._session.flush()
        return paper_version_record_to_domain(record)

    def update_version_parse_result(
        self,
        paper_version_id: str,
        *,
        parsed_artifact_path: str,
        parsed_at: datetime,
        parser_name: str,
        parser_version: str,
        page_count: int,
        section_count: int,
        warning_count: int,
    ) -> PaperVersion:
        """Persist parsed-document metadata onto an existing paper version.

        Same shape as `update_version_artifact`: only ever called by
        `PaperParsingService` after a real parse or artifact reconciliation
        has determined the correct values.
        """

        record = self._session.get(PaperVersionRecord, paper_version_id)
        if record is None:
            raise PaperVersionNotFoundError(f"paper version {paper_version_id} not found")

        record.parsed_artifact_path = parsed_artifact_path
        record.parsed_at = parsed_at
        record.parser_name = parser_name
        record.parser_version = parser_version
        record.page_count = page_count
        record.section_count = section_count
        record.warning_count = warning_count
        self._session.flush()
        return paper_version_record_to_domain(record)

    def update_version_chunk_result(
        self,
        paper_version_id: str,
        *,
        chunked_artifact_path: str,
        chunked_at: datetime,
        chunk_count: int,
        chunking_version: str,
        chunk_artifact_checksum: str,
        chunk_config_fingerprint: str,
    ) -> PaperVersion:
        """Persist chunk-artifact metadata onto an existing paper version.

        Same shape as `update_version_parse_result`: only ever called by
        `ChunkingService` after a real chunk run or artifact reconciliation
        has determined the correct values. `chunk_config_fingerprint`
        (prompt 6.1) is the actual reuse/invalidation identity;
        `chunking_version` remains for readability/debugging only.
        """

        record = self._session.get(PaperVersionRecord, paper_version_id)
        if record is None:
            raise PaperVersionNotFoundError(f"paper version {paper_version_id} not found")

        record.chunked_artifact_path = chunked_artifact_path
        record.chunked_at = chunked_at
        record.chunk_count = chunk_count
        record.chunking_version = chunking_version
        record.chunk_artifact_checksum = chunk_artifact_checksum
        record.chunk_config_fingerprint = chunk_config_fingerprint
        self._session.flush()
        return paper_version_record_to_domain(record)

    def update_version_vector_result(
        self,
        paper_version_id: str,
        *,
        vector_indexed_at: datetime,
        vector_count: int,
        embedding_provider: str,
        embedding_model: str,
        embedding_config_fingerprint: str,
        vector_generation_fingerprint: str,
        qdrant_collection: str,
    ) -> PaperVersion:
        """Persist vector-index metadata onto an existing paper version.

        Same shape as `update_version_chunk_result`: only ever called by
        `VectorIndexingService` after a real indexing run or Qdrant
        reconciliation has determined the correct values.
        `vector_generation_fingerprint` is the actual reuse/invalidation
        identity; `embedding_provider`/`embedding_model` remain for
        readability/debugging only.
        """

        record = self._session.get(PaperVersionRecord, paper_version_id)
        if record is None:
            raise PaperVersionNotFoundError(f"paper version {paper_version_id} not found")

        record.vector_indexed_at = vector_indexed_at
        record.vector_count = vector_count
        record.embedding_provider = embedding_provider
        record.embedding_model = embedding_model
        record.embedding_config_fingerprint = embedding_config_fingerprint
        record.vector_generation_fingerprint = vector_generation_fingerprint
        record.qdrant_collection = qdrant_collection
        self._session.flush()
        return paper_version_record_to_domain(record)

    def update_version_graph_extraction_result(
        self,
        paper_version_id: str,
        *,
        graph_extraction_artifact_path: str,
        graph_extracted_at: datetime,
        entity_count: int,
        relationship_count: int,
        extraction_version: str,
        extraction_config_fingerprint: str,
        graph_extraction_generation_fingerprint: str,
        graph_extraction_artifact_checksum: str,
    ) -> PaperVersion:
        """Persist graph-extraction metadata onto an existing paper version.

        Same shape as `update_version_vector_result`: only ever called by
        `ScientificKnowledgeExtractionService` after a real extraction run
        or artifact reconciliation has determined the correct values.
        `graph_extraction_generation_fingerprint` is the actual
        reuse/invalidation identity; `extraction_version` remains for
        readability/debugging only. Never writes to Neo4j -- this is
        extraction-artifact metadata only.
        """

        record = self._session.get(PaperVersionRecord, paper_version_id)
        if record is None:
            raise PaperVersionNotFoundError(f"paper version {paper_version_id} not found")

        record.graph_extraction_artifact_path = graph_extraction_artifact_path
        record.graph_extracted_at = graph_extracted_at
        record.entity_count = entity_count
        record.relationship_count = relationship_count
        record.extraction_version = extraction_version
        record.extraction_config_fingerprint = extraction_config_fingerprint
        record.graph_extraction_generation_fingerprint = graph_extraction_generation_fingerprint
        record.graph_extraction_artifact_checksum = graph_extraction_artifact_checksum
        self._session.flush()
        return paper_version_record_to_domain(record)

    def update_version_graph_index_result(
        self,
        paper_version_id: str,
        *,
        graph_indexed_at: datetime,
        canonical_entity_count: int,
        graph_relationship_count: int,
        canonicalization_config_fingerprint: str,
        graph_index_generation_fingerprint: str,
        neo4j_database: str,
    ) -> PaperVersion:
        """Persist graph-index metadata onto an existing paper version.

        Same shape as `update_version_graph_extraction_result`: only ever
        called by `GraphIndexingService` after a real indexing run or
        Neo4j reconciliation has determined the correct values.
        `graph_index_generation_fingerprint` is the actual
        reuse/invalidation identity. The graph itself is never stored
        here -- Neo4j is the only place node/relationship data lives.
        """

        record = self._session.get(PaperVersionRecord, paper_version_id)
        if record is None:
            raise PaperVersionNotFoundError(f"paper version {paper_version_id} not found")

        record.graph_indexed_at = graph_indexed_at
        record.canonical_entity_count = canonical_entity_count
        record.graph_relationship_count = graph_relationship_count
        record.canonicalization_config_fingerprint = canonicalization_config_fingerprint
        record.graph_index_generation_fingerprint = graph_index_generation_fingerprint
        record.neo4j_database = neo4j_database
        self._session.flush()
        return paper_version_record_to_domain(record)
