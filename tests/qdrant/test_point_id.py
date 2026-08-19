"""Unit tests for `build_qdrant_point_id` (prompt #16)."""

import uuid

from app.storage.qdrant.models import build_qdrant_point_id


class TestQdrantPointId:
    def test_same_chunk_id_produces_the_same_point_id(self) -> None:
        assert build_qdrant_point_id("chunk:abc123") == build_qdrant_point_id("chunk:abc123")

    def test_different_chunk_id_produces_a_different_point_id(self) -> None:
        assert build_qdrant_point_id("chunk:abc123") != build_qdrant_point_id("chunk:def456")

    def test_result_is_a_valid_uuid(self) -> None:
        point_id = build_qdrant_point_id("chunk:abc123")
        uuid.UUID(point_id)  # must not raise

    def test_stable_across_repeated_calls_not_random(self) -> None:
        # Never a random UUID (prompt #16) -- ten calls, one distinct value.
        results = {build_qdrant_point_id("chunk:stable") for _ in range(10)}
        assert len(results) == 1
