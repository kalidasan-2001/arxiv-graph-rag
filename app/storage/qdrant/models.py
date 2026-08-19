"""DTOs and identity helpers for the Qdrant vector store adapter.

`qdrant_client.models.PointStruct` (and every other vendor SDK class) stops
at `qdrant_repository.py` -- everything above that layer works with these
plain, provider-independent types instead (CLAUDE.md #15).
"""

import uuid

from pydantic import BaseModel

# Fixed, checked-in constant -- point ids must be deterministic functions of
# `chunk_id` (prompt #16: "do not use random UUIDs"), and UUID5 with a fixed
# namespace is exactly that: the same `chunk_id` always derives the same
# point id, in this process or any other, forever.
_QDRANT_POINT_NAMESPACE = uuid.UUID("6e2f6a3e-6b1e-4b8b-9c1e-3b7b7e9f9a10")


def build_qdrant_point_id(chunk_id: str) -> str:
    """Derive a Qdrant-compatible point id from a stable `chunk_id`.

    Qdrant point ids must be an unsigned integer or a UUID (prompt #16) --
    `chunk_id` (e.g. `"chunk:3a99b6625c7b2ded"`) is neither, so this
    deterministically maps it into UUID space instead of generating a
    random one. `chunk_id` itself is always kept in the point's payload,
    so the original domain identity is never lost.
    """

    return str(uuid.uuid5(_QDRANT_POINT_NAMESPACE, chunk_id))


class VectorPointPayload(BaseModel):
    """Everything retrieval/citations need about one indexed chunk
    (prompt #17) -- the full chunk text is included (prompt #17/#50): at
    this project's controlled scale, storing it here is a deliberate
    simplification that avoids a second document-store round trip for
    every retrieved chunk, not a duplicated "whole paper"."""

    chunk_id: str
    paper_id: str
    paper_version_id: str
    section_id: str
    section_type: str
    section_title: str | None
    chunk_index: int
    page_start: int | None
    page_end: int | None
    source: str
    source_id: str
    published_year: int | None
    categories: list[str]
    chunking_version: str
    chunk_config_fingerprint: str
    embedding_provider: str
    embedding_model: str
    embedding_config_fingerprint: str
    vector_generation_fingerprint: str
    text: str


class VectorPoint(BaseModel):
    """One point ready to upsert: a Qdrant-compatible id, its vector, and
    its full payload."""

    point_id: str
    vector: list[float]
    payload: VectorPointPayload


class VectorSearchHit(BaseModel):
    """One ranked semantic-search result (prompt #36)."""

    chunk_id: str
    paper_id: str
    paper_version_id: str
    section_id: str
    section_type: str
    section_title: str | None
    chunk_index: int
    page_start: int | None
    page_end: int | None
    text: str
    vector_generation_fingerprint: str
    # Never called "probability"/"confidence" (prompt #40) -- it's exactly
    # what the configured Qdrant distance metric reports, no more.
    similarity_score: float


class VectorChunkRecord(BaseModel):
    """One exact Qdrant payload record resolved by stable chunk identity."""

    chunk_id: str
    paper_id: str
    paper_version_id: str
    section_id: str
    section_type: str
    section_title: str | None
    chunk_index: int
    page_start: int | None
    page_end: int | None
    text: str
    vector_generation_fingerprint: str
