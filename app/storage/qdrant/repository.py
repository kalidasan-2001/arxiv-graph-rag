"""The vector storage abstraction the rest of the application depends on.

Mirrors `app.ingestion.chunking.chunker.ScientificChunker`: application
code depends only on this `Protocol`, never on `qdrant_client` directly
(CLAUDE.md #15). Only the methods `VectorIndexingService` and the semantic
search endpoint actually need (prompt #15) -- not a general vector-database
framework.
"""

from typing import Protocol

from app.storage.qdrant.models import VectorChunkRecord, VectorPoint, VectorSearchHit


class VectorRepository(Protocol):
    """Storage operations for indexed paper chunks."""

    def ensure_collection(self, *, dimension: int, distance: str) -> None:
        """Create the configured collection if missing; if it already
        exists, verify its dimension/distance match `dimension`/`distance`
        exactly. Raises `VectorCollectionIncompatibleError` on a mismatch
        (prompt #14) -- never silently deletes/recreates it."""
        ...

    def upsert_chunks(self, points: list[VectorPoint]) -> None:
        """Insert or overwrite `points` by their (deterministic) point id.
        Idempotent -- retrying with the same points is always safe (prompt #24)."""
        ...

    def count_for_paper_version(
        self, paper_version_id: str, *, generation_fingerprint: str | None = None
    ) -> int:
        """Count points for `paper_version_id`, optionally filtered to one
        `vector_generation_fingerprint` (prompt #31) -- the reconciliation
        primitive: comparing this against the expected chunk count is how
        VALID/MISSING/PARTIAL/STALE is distinguished (prompt #30)."""
        ...

    def delete_paper_version(
        self, paper_version_id: str, *, exclude_generation_fingerprint: str | None = None
    ) -> int:
        """Delete points for `paper_version_id`, optionally keeping points
        that belong to `exclude_generation_fingerprint` (prompt #32/#33) --
        used to remove a stale generation after its replacement has been
        verified complete. Returns the number of points deleted. Always
        scoped to one `paper_version_id`; never touches other papers'
        vectors or recreates the collection (prompt #33)."""
        ...

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        paper_id: str | None = None,
        paper_version_id: str | None = None,
        section_type: str | None = None,
    ) -> list[VectorSearchHit]:
        """Ranked nearest-neighbor search, optionally filtered (prompt #38).
        References are never excluded automatically (prompt #42) -- callers
        filter by `section_type` explicitly if they want that."""
        ...

    def get_by_chunk_ids(self, chunk_ids: list[str]) -> list[VectorChunkRecord]:
        """Exact identity lookup by Qdrant payload `chunk_id`.

        This is for provenance bridging, not semantic search: no nearest
        neighbor query is performed and missing chunks are simply absent
        from the returned list.
        """
        ...
