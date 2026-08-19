"""`VectorRepository` implementation backed by `qdrant-client`.

The only module in the codebase that imports `qdrant_client` directly
(besides `client.py`'s connection factory) -- everything above this layer
works with `app.storage.qdrant.models` DTOs (CLAUDE.md #15).
"""

import logging

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException
from qdrant_client.http import models as qmodels

from app.core.config import get_settings
from app.core.exceptions import VectorCollectionIncompatibleError, VectorStoreUnavailableError
from app.storage.qdrant.client import get_qdrant_client
from app.storage.qdrant.models import VectorChunkRecord, VectorPoint, VectorSearchHit

logger = logging.getLogger(__name__)

# V1 supports exactly one metric, chosen for normalized sentence-transformer
# embeddings (prompt #13) -- not user-configurable yet, since nothing in
# this project needs a second one.
_DISTANCE_MAP = {"cosine": qmodels.Distance.COSINE}


class QdrantVectorRepository:
    """Talks to one configured Qdrant collection."""

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection = collection_name

    def ensure_collection(self, *, dimension: int, distance: str) -> None:
        target_distance = _DISTANCE_MAP[distance]
        try:
            exists = self._client.collection_exists(self._collection)
        except ApiException as exc:
            raise VectorStoreUnavailableError(f"could not reach Qdrant: {exc}") from exc

        if not exists:
            try:
                self._client.create_collection(
                    self._collection,
                    vectors_config=qmodels.VectorParams(size=dimension, distance=target_distance),
                )
            except ApiException as exc:
                raise VectorStoreUnavailableError(
                    f"failed to create Qdrant collection {self._collection!r}: {exc}"
                ) from exc
            logger.info(
                "created Qdrant collection name=%s dimension=%d distance=%s status=ok",
                self._collection,
                dimension,
                distance,
            )
            return

        try:
            info = self._client.get_collection(self._collection)
        except ApiException as exc:
            raise VectorStoreUnavailableError(f"could not reach Qdrant: {exc}") from exc

        existing = info.config.params.vectors
        if isinstance(existing, dict):
            # Named-vector collections are a different Qdrant configuration
            # this application never creates -- treat as incompatible
            # rather than guessing which named vector to use.
            raise VectorCollectionIncompatibleError(
                f"Qdrant collection {self._collection!r} uses named vectors, "
                "which this application does not support"
            )
        if existing.size != dimension or existing.distance != target_distance:
            raise VectorCollectionIncompatibleError(
                f"Qdrant collection {self._collection!r} has dimension={existing.size} "
                f"distance={existing.distance.value}; the active embedding provider needs "
                f"dimension={dimension} distance={distance}. Never auto-deleted/recreated -- "
                "use a differently-named collection or an explicit rebuild if this is intentional."
            )

    def upsert_chunks(self, points: list[VectorPoint]) -> None:
        if not points:
            return
        try:
            self._client.upsert(
                self._collection,
                points=[
                    qmodels.PointStruct(
                        id=point.point_id,
                        vector=point.vector,
                        payload=point.payload.model_dump(mode="json"),
                    )
                    for point in points
                ],
            )
        except ApiException as exc:
            raise VectorStoreUnavailableError(f"Qdrant upsert failed: {exc}") from exc

    def count_for_paper_version(
        self, paper_version_id: str, *, generation_fingerprint: str | None = None
    ) -> int:
        conditions = [
            qmodels.FieldCondition(
                key="paper_version_id", match=qmodels.MatchValue(value=paper_version_id)
            )
        ]
        if generation_fingerprint is not None:
            conditions.append(
                qmodels.FieldCondition(
                    key="vector_generation_fingerprint",
                    match=qmodels.MatchValue(value=generation_fingerprint),
                )
            )
        try:
            result = self._client.count(
                self._collection, count_filter=qmodels.Filter(must=conditions), exact=True
            )
        except ApiException as exc:
            raise VectorStoreUnavailableError(f"Qdrant count failed: {exc}") from exc
        return result.count

    def delete_paper_version(
        self, paper_version_id: str, *, exclude_generation_fingerprint: str | None = None
    ) -> int:
        must = [
            qmodels.FieldCondition(
                key="paper_version_id", match=qmodels.MatchValue(value=paper_version_id)
            )
        ]
        must_not = []
        if exclude_generation_fingerprint is not None:
            must_not.append(
                qmodels.FieldCondition(
                    key="vector_generation_fingerprint",
                    match=qmodels.MatchValue(value=exclude_generation_fingerprint),
                )
            )
        delete_filter = qmodels.Filter(must=must, must_not=must_not)

        # Deletion is always scoped to exactly one paper_version_id (prompt
        # #33) -- count first (cheap, exact) so the caller/logs know how
        # many stale points this operation actually removed.
        before = self.count_for_paper_version(paper_version_id)
        if before == 0:
            return 0
        try:
            self._client.delete(
                self._collection, points_selector=qmodels.FilterSelector(filter=delete_filter)
            )
        except ApiException as exc:
            raise VectorStoreUnavailableError(f"Qdrant delete failed: {exc}") from exc
        after = self.count_for_paper_version(paper_version_id)
        return before - after

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        paper_id: str | None = None,
        paper_version_id: str | None = None,
        section_type: str | None = None,
    ) -> list[VectorSearchHit]:
        conditions = []
        if paper_id is not None:
            conditions.append(
                qmodels.FieldCondition(key="paper_id", match=qmodels.MatchValue(value=paper_id))
            )
        if paper_version_id is not None:
            conditions.append(
                qmodels.FieldCondition(
                    key="paper_version_id", match=qmodels.MatchValue(value=paper_version_id)
                )
            )
        if section_type is not None:
            conditions.append(
                qmodels.FieldCondition(
                    key="section_type", match=qmodels.MatchValue(value=section_type)
                )
            )
        query_filter = qmodels.Filter(must=conditions) if conditions else None

        try:
            response = self._client.query_points(
                self._collection,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        except ApiException as exc:
            raise VectorStoreUnavailableError(f"Qdrant search failed: {exc}") from exc

        return [
            VectorSearchHit(
                chunk_id=point.payload["chunk_id"],
                paper_id=point.payload["paper_id"],
                paper_version_id=point.payload["paper_version_id"],
                section_id=point.payload["section_id"],
                section_type=point.payload["section_type"],
                section_title=point.payload.get("section_title"),
                chunk_index=point.payload["chunk_index"],
                page_start=point.payload.get("page_start"),
                page_end=point.payload.get("page_end"),
                text=point.payload["text"],
                vector_generation_fingerprint=point.payload["vector_generation_fingerprint"],
                similarity_score=point.score,
            )
            for point in response.points
        ]

    def get_by_chunk_ids(self, chunk_ids: list[str]) -> list[VectorChunkRecord]:
        unique_chunk_ids = sorted(set(chunk_ids))
        if not unique_chunk_ids:
            return []
        lookup_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="chunk_id", match=qmodels.MatchAny(any=unique_chunk_ids)
                )
            ]
        )
        try:
            scrolled_points, _offset = self._client.scroll(
                self._collection,
                scroll_filter=lookup_filter,
                limit=len(unique_chunk_ids),
                with_payload=True,
                with_vectors=False,
            )
        except ApiException as exc:
            raise VectorStoreUnavailableError(f"Qdrant exact chunk lookup failed: {exc}") from exc

        records = [
            VectorChunkRecord(
                chunk_id=point.payload["chunk_id"],
                paper_id=point.payload["paper_id"],
                paper_version_id=point.payload["paper_version_id"],
                section_id=point.payload["section_id"],
                section_type=point.payload["section_type"],
                section_title=point.payload.get("section_title"),
                chunk_index=point.payload["chunk_index"],
                page_start=point.payload.get("page_start"),
                page_end=point.payload.get("page_end"),
                text=point.payload["text"],
                vector_generation_fingerprint=point.payload["vector_generation_fingerprint"],
            )
            for point in scrolled_points
        ]
        return sorted(records, key=lambda record: record.chunk_id)


def get_vector_repository() -> "QdrantVectorRepository":
    """FastAPI dependency (mirrors `get_embedding_provider`/`get_arxiv_client`
    -- no arguments, reads `Settings` itself) so tests can override it via
    `app.dependency_overrides` instead of a real Qdrant instance."""

    settings = get_settings()
    return QdrantVectorRepository(get_qdrant_client(settings), settings.QDRANT_COLLECTION)
