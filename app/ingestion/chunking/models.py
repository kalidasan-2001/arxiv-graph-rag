"""DTOs for the chunking stage: chunking configuration, diagnostics, and
the resulting structured chunk artifact.

`ChunkedPaperDocument.chunks` reuses the existing domain `PaperChunk`
model directly -- no new identity system (CLAUDE.md; prompt #9).
"""

from enum import Enum

from pydantic import BaseModel, Field

from app.domain.papers import PaperChunk


class ChunkWarning(str, Enum):
    """Deterministic, non-fatal chunking quality signals (prompt #39)."""

    EMPTY_SECTION_SKIPPED = "empty_section_skipped"
    TINY_FRAGMENT_MERGED = "tiny_fragment_merged"
    OVERSIZED_CHUNK = "oversized_chunk"
    PAGE_PROVENANCE_APPROXIMATE = "page_provenance_approximate"
    REFERENCES_PRESERVED = "references_preserved"


class ChunkingConfig(BaseModel):
    """Exactly what produced this chunk artifact -- the identity surface
    reparse-invalidation compares against (prompt #7/#29).

    `config_fingerprint` (prompt 6.1) is a SHA-256 of every field below
    that materially affects chunk output (see
    `build_chunk_config_fingerprint`) and is what reuse/invalidation
    actually compares -- `version` alone is a human-maintained label that
    nothing forces a developer to bump when e.g. `chunk_size_tokens`
    changes, so it is informational here, not the identity check.

    Deliberately has no default for `config_fingerprint`: a `chunks.json`
    written before this field existed (a "legacy artifact") fails Pydantic
    validation on read rather than being silently treated as valid --
    `ChunkArtifactStorage.try_read()` already maps any validation failure
    to "no valid artifact", which safely forces a real rechunk.
    """

    version: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    min_chunk_tokens: int
    tokenizer: str
    tokenizer_version: str | None = None
    config_fingerprint: str


class ChunkDiagnostics(BaseModel):
    """Deterministic chunk-size diagnostics (prompt #37)."""

    chunk_count: int
    min_tokens: int
    max_tokens: int
    average_tokens: float
    median_tokens: float
    small_chunk_count: int
    oversized_chunk_count: int


class ChunkedPaperDocument(BaseModel):
    """The full structured output of chunking one paper version.

    `source_pdf_checksum` and `parsed_artifact_checksum` are distinct
    fields (prompt #23: never overload one checksum field for two
    different artifacts) -- both are filled in by `ChunkingService` after
    the chunker itself runs, since the chunker operates on an in-memory
    `ParsedPaperDocument` and doesn't know about file checksums.
    """

    paper_id: str
    paper_version_id: str
    source_pdf_checksum: str = ""
    parsed_artifact_checksum: str = ""
    chunking: ChunkingConfig
    chunks: list[PaperChunk] = Field(default_factory=list)
    diagnostics: ChunkDiagnostics
    warnings: list[ChunkWarning] = Field(default_factory=list)
