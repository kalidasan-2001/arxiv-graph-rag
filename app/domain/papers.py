"""Paper, revision, section, and chunk domain models.

A logical ``Paper`` is separated from ``PaperVersion`` because scientific
papers are revised (arXiv v1, v2, v3, ...) without becoming a different
paper. Everything downstream (sections, chunks) is scoped to a specific
version, since re-parsing a new revision must not silently overwrite or
mix content from a prior one.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.domain.enums import SectionType
from app.domain.ids import (
    build_chunk_id,
    build_paper_id,
    build_paper_version_id,
    build_section_id,
    ensure_json_safe,
    normalize_whitespace,
)


class Paper(BaseModel):
    """A logical scientific paper, independent of any specific revision.

    ``source`` identifies where the paper came from (``"arxiv"`` today);
    the model deliberately does not assume every future source has an
    arXiv-shaped id, so ``source_id`` is an opaque string.
    """

    paper_id: str
    source: str
    source_id: str
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    published_at: datetime | None = None
    updated_at: datetime | None = None
    pdf_url: str | None = None

    @field_validator("source", "source_id", "title")
    @classmethod
    def _required_text(cls, value: str) -> str:
        normalized = normalize_whitespace(value)
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator("abstract")
    @classmethod
    def _normalize_abstract(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_whitespace(value)

    @field_validator("authors", "categories")
    @classmethod
    def _normalize_string_list(cls, values: list[str]) -> list[str]:
        return [normalized for v in values if (normalized := normalize_whitespace(v))]

    @classmethod
    def create(cls, *, source: str, source_id: str, title: str, **kwargs: Any) -> "Paper":
        """Construct a ``Paper`` with its id derived via `build_paper_id`."""

        return cls(
            paper_id=build_paper_id(source, source_id),
            source=source,
            source_id=source_id,
            title=title,
            **kwargs,
        )


class PaperVersion(BaseModel):
    """A specific, immutable revision of a ``Paper`` (e.g. arXiv v2).

    ``checksum`` is always a SHA-256 hex digest of the stored raw PDF
    artifact (the schema stores one checksum string, so this is documented
    rather than tracked in a separate ``checksum_algorithm`` column).
    ``storage_path``/``downloaded_at``/``file_size_bytes`` are populated
    once the artifact has actually been downloaded (Prompt 4) -- all three
    are ``None`` for a version that is discovered but not yet ingested.

    ``parsed_artifact_path``/``parsed_at``/``parser_name``/``parser_version``/
    ``page_count``/``section_count``/``warning_count`` are populated once
    the PDF has been parsed (Prompt 5); all are ``None`` before that. Note
    ``parser_version`` already existed as an unused placeholder from
    Prompt 1 -- Prompt 5 is what actually populates it.

    ``chunked_artifact_path``/``chunked_at``/``chunk_count``/
    ``chunking_version``/``chunk_artifact_checksum`` are populated once the
    parsed document has been chunked (Prompt 6); all are ``None`` before
    that. ``chunk_config_fingerprint`` (Prompt 6.1) is the deterministic
    fingerprint of the *complete* effective chunking configuration that
    produced those chunks -- reuse/invalidation checks compare this, not
    ``chunking_version`` alone, since nothing forces a version bump when
    e.g. chunk size or overlap changes.

    ``vector_indexed_at``/``vector_count``/``embedding_provider``/
    ``embedding_model``/``embedding_config_fingerprint``/
    ``vector_generation_fingerprint``/``qdrant_collection`` are populated
    once the chunk artifact has been embedded and indexed into Qdrant
    (Prompt 7); all are ``None`` before that. ``vector_generation_fingerprint``
    (derived from ``chunk_artifact_checksum`` + ``embedding_config_fingerprint``)
    is the actual reuse/invalidation identity -- see
    ``app.ingestion.vector_indexing.fingerprint``.

    ``graph_extraction_artifact_path``/``graph_extracted_at``/
    ``entity_count``/``relationship_count``/``extraction_version``/
    ``extraction_config_fingerprint``/``graph_extraction_generation_fingerprint``/
    ``graph_extraction_artifact_checksum`` are populated once the chunk
    artifact has been extracted into scientific entities/relationships
    (Prompt 8); all are ``None`` before that. This is extraction only --
    Neo4j is not written to yet, so none of these fields imply a knowledge
    graph exists; see ``app.ingestion.graph_extraction.fingerprint``.

    ``graph_indexed_at``/``canonical_entity_count``/``graph_relationship_count``/
    ``canonicalization_config_fingerprint``/``graph_index_generation_fingerprint``/
    ``neo4j_database`` are populated once the extraction artifact has been
    canonicalized and written to Neo4j (Prompt 9); all are ``None`` before
    that. The graph itself is never stored here -- only enough metadata to
    answer "which Neo4j generation corresponds to this paper version";
    see ``app.ingestion.canonical_resolution.fingerprint``.
    """

    paper_version_id: str
    paper_id: str
    version: str
    source_version: str | None = None
    checksum: str | None = None
    parser_version: str | None = None
    storage_path: str | None = None
    downloaded_at: datetime | None = None
    file_size_bytes: int | None = None
    parsed_artifact_path: str | None = None
    parsed_at: datetime | None = None
    parser_name: str | None = None
    page_count: int | None = None
    section_count: int | None = None
    warning_count: int | None = None
    chunked_artifact_path: str | None = None
    chunked_at: datetime | None = None
    chunk_count: int | None = None
    chunking_version: str | None = None
    chunk_artifact_checksum: str | None = None
    chunk_config_fingerprint: str | None = None
    vector_indexed_at: datetime | None = None
    vector_count: int | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_config_fingerprint: str | None = None
    vector_generation_fingerprint: str | None = None
    qdrant_collection: str | None = None
    graph_extraction_artifact_path: str | None = None
    graph_extracted_at: datetime | None = None
    entity_count: int | None = None
    relationship_count: int | None = None
    extraction_version: str | None = None
    extraction_config_fingerprint: str | None = None
    graph_extraction_generation_fingerprint: str | None = None
    graph_extraction_artifact_checksum: str | None = None
    graph_indexed_at: datetime | None = None
    canonical_entity_count: int | None = None
    graph_relationship_count: int | None = None
    canonicalization_config_fingerprint: str | None = None
    graph_index_generation_fingerprint: str | None = None
    neo4j_database: str | None = None
    created_at: datetime | None = None

    @field_validator("version")
    @classmethod
    def _version_required(cls, value: str) -> str:
        normalized = normalize_whitespace(value)
        if not normalized:
            raise ValueError("version must not be blank")
        return normalized

    @field_validator(
        "file_size_bytes", "page_count", "section_count", "warning_count", "chunk_count",
        "vector_count", "entity_count", "relationship_count",
        "canonical_entity_count", "graph_relationship_count",
    )
    @classmethod
    def _non_negative_if_set(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("value must be >= 0")
        return value

    @classmethod
    def create(cls, *, paper_id: str, version: str, **kwargs: Any) -> "PaperVersion":
        """Construct a ``PaperVersion`` with its id derived via `build_paper_version_id`."""

        return cls(
            paper_version_id=build_paper_version_id(paper_id, version),
            paper_id=paper_id,
            version=version,
            **kwargs,
        )


class PaperSection(BaseModel):
    """A structural section of a specific paper version (e.g. "Methodology")."""

    section_id: str
    paper_id: str
    paper_version_id: str
    section_type: SectionType
    title: str | None = None
    order: int
    page_start: int | None = None
    page_end: int | None = None
    text: str

    @field_validator("order")
    @classmethod
    def _order_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("order must be >= 0")
        return value

    @field_validator("text")
    @classmethod
    def _text_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    @model_validator(mode="after")
    def _pages_ordered(self) -> "PaperSection":
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_start > self.page_end
        ):
            raise ValueError("page_start must be <= page_end")
        return self

    @classmethod
    def create(
        cls,
        *,
        paper_id: str,
        paper_version_id: str,
        section_type: SectionType,
        order: int,
        text: str,
        **kwargs: Any,
    ) -> "PaperSection":
        """Construct a ``PaperSection`` with its id derived via `build_section_id`."""

        return cls(
            section_id=build_section_id(paper_version_id, section_type, order),
            paper_id=paper_id,
            paper_version_id=paper_version_id,
            section_type=section_type,
            order=order,
            text=text,
            **kwargs,
        )


class PaperChunk(BaseModel):
    """A retrieval-sized slice of a section's text.

    Never carries an embedding or a vendor SDK object -- those belong to
    the future Qdrant repository layer, not the domain model.
    """

    chunk_id: str
    paper_id: str
    paper_version_id: str
    section_id: str
    section_type: SectionType
    chunk_index: int
    text: str
    token_count: int
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunk_index", "token_count")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("value must be >= 0")
        return value

    @field_validator("text")
    @classmethod
    def _text_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    @field_validator("metadata")
    @classmethod
    def _metadata_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_json_safe(value)

    @model_validator(mode="after")
    def _pages_ordered(self) -> "PaperChunk":
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_start > self.page_end
        ):
            raise ValueError("page_start must be <= page_end")
        return self

    @classmethod
    def create(
        cls,
        *,
        paper_id: str,
        paper_version_id: str,
        section_id: str,
        section_type: SectionType,
        chunk_index: int,
        text: str,
        token_count: int,
        chunk_config_fingerprint: str,
        **kwargs: Any,
    ) -> "PaperChunk":
        """Construct a ``PaperChunk`` with its id derived via `build_chunk_id`.

        `chunk_config_fingerprint` (prompt 6.1) -- not a bare chunking
        version string -- so that any materially different chunking
        configuration (size, overlap, minimum size, tokenizer, or version)
        produces distinguishable chunk ids; see `build_chunk_id`.
        """

        return cls(
            chunk_id=build_chunk_id(
                paper_version_id, section_id, chunk_index, chunk_config_fingerprint
            ),
            paper_id=paper_id,
            paper_version_id=paper_version_id,
            section_id=section_id,
            section_type=section_type,
            chunk_index=chunk_index,
            text=text,
            token_count=token_count,
            **kwargs,
        )
