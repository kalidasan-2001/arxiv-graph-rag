"""Minimal semantic vector search (prompt #36) -- validates the vector
indexing layer directly. Not RAG: no LLM, no synthesis, just ranked chunks.
"""

from app.core.exceptions import VectorSearchError
from app.embeddings.provider import EmbeddingProvider
from app.storage.qdrant.models import VectorSearchHit
from app.storage.qdrant.repository import VectorRepository


class VectorSearchService:
    """Query embedding -> Qdrant -> ranked `PaperChunk`s. Deterministic and
    directly testable, independent of any future reasoning layer
    (CLAUDE.md #21)."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_repository: VectorRepository,
        *,
        default_top_k: int,
        max_top_k: int,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_repo = vector_repository
        self._default_top_k = default_top_k
        self._max_top_k = max_top_k

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        paper_id: str | None = None,
        paper_version_id: str | None = None,
        section_type: str | None = None,
    ) -> list[VectorSearchHit]:
        if not query.strip():
            raise VectorSearchError("query must not be blank")

        resolved_top_k = self._default_top_k if top_k is None else top_k
        if resolved_top_k <= 0 or resolved_top_k > self._max_top_k:
            raise VectorSearchError(
                f"top_k must be between 1 and {self._max_top_k} (got {resolved_top_k})"
            )

        # `embed_query`, not `embed_documents` (prompt #37) -- some models
        # use a different prompt/prefix for queries vs. indexed passages.
        query_vector = self._embedding_provider.embed_query(query)

        return self._vector_repo.search(
            query_vector,
            top_k=resolved_top_k,
            paper_id=paper_id,
            paper_version_id=paper_version_id,
            section_type=section_type,
        )
